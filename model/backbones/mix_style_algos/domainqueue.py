import random
import torch
import torch.nn as nn


class DomainQueue(nn.Module):


    def __init__(self, num_features, num_domains, p=0.5, alpha=0.1, eps=1e-6, mix='random', capacity=1024):
        """
        Args:
          p (float): probability of using mix.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        
        self.sum = list(0 for _ in range(num_domains))
        self.capacity = capacity
        self.register_buffer('mean_queue', torch.zeros(num_domains, capacity, num_features))
        self.register_buffer('sig_queue', torch.ones(num_domains, capacity, num_features))
        self.mean_queue.requires_grad = False
        self.sig_queue.requires_grad = False

    def forward(self, x, domain=None):
        if not self.training:
            return x

        for i in range(self.num_domains):
            if not i in domain:
                continue
            mean = x[domain==i].mean(1).detach()
            sig = (x[domain==i].var(1)+self.eps).sqrt().detach()
            length = (domain==i).sum()

            sum = self.sum[i] % self.capacity
            rest = self.capacity - sum
            if length > rest:
                self.mean_queue[i, sum:] = mean[:rest]
                self.mean_queue[i, :length-rest] = mean[rest:]
                self.sig_queue[i, sum:] = sig[:rest]
                self.sig_queue[i, :length-rest] = sig[rest:]
            else:
                self.mean_queue[i, sum:sum+length] = mean
                self.sig_queue[i, sum:sum+length] = sig
            self.sum[i] = self.sum[i] + int(length)

        if not random.random() > self.p:
            return x
        
        B = x.size(0)
        mu = x.mean(dim=1, keepdim=True) ### origin dim=1
        var = x.var(dim=1, keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)

        # #### random indexes (1 ~ num domains - 1)
        # #### make sure that inds are all different from domain
        # d_ind1 = random.choices(range(1, self.num_domains), k=B)
        # d_ind1 = torch.tensor(d_ind1, device=domain.device) + domain
        # d_ind1 = d_ind1 % self.num_domains
        # # d_ind2 = random.choices(range(1, self.num_domains), k=B)
        # # d_ind2 = torch.tensor(d_ind2, device=domain.device) + domain
        # # d_ind2 = d_ind2 % self.num_domains
        # # f_inds = torch.randint_like(domain, self.capacity)
        # f_ind1 = torch.tensor([random.randint(0, self.sum[d_ind1[i]] % self.capacity) for i in range(B)])
        # # f_ind2 = torch.tensor([random.randint(0, self.sum[d_ind1[i]] % self.capacity) for i in range(B)])
        # mu1 = self.mean_queue[d_ind1, f_ind1].unsqueeze(1)
        # sig1 = self.sig_queue[d_ind1, f_ind1].unsqueeze(1)
        # # mu2 = self.mean_queue[d_ind2, f_ind2].unsqueeze(1)
        # # sig2 = self.sig_queue[d_ind2, f_ind2].unsqueeze(1)

        #### fuse all domains
        idxs = torch.tensor([[random.randint(0, self.sum[i] % self.capacity) for _ in range(B)]for i in range(self.num_domains)]).to(x.device)
        ratios = torch.rand([self.num_domains,B])
        ratios = ratios.softmax(dim=0).to(x.device)
        mu1 = torch.zeros([B,self.num_features], dtype=x.dtype).to(x.device)
        sig1 = torch.zeros([B,self.num_features], dtype=x.dtype).to(x.device)
        for i in range(self.num_domains):
            mu1 = mu1 + self.mean_queue[i].index_select(0, idxs[i]) * ratios[i].unsqueeze(1)
            sig1 = sig1 + self.sig_queue[i].index_select(0, idxs[i]) * ratios[i].unsqueeze(1)
        mu1 = mu1.unsqueeze(1) / self.num_domains
        sig1 = sig1.unsqueeze(1) / self.num_domains


        # #### equal ratio of mu, sig formation (just one mu, sig)
        # dom_list = list()
        # for i in range(self.num_domains):
        #     if self.sum[i] != 0:
        #         dom_list.append(i)
        # k_dom = random.randint(1, len(dom_list))
        # dom_ind = random.choices(dom_list, k=k_dom)
        # mu_, sig_ = torch.zeros_like(mu), torch.zeros_like(sig)
        
        # for d_ind in dom_ind:
        #     f_ind = random.choice(range(0, min(self.sum[d_ind], self.capacity)))
        #     mu_ = mu_ + self.mean_queue[d_ind, f_ind]
        #     sig_ = sig_ + self.sig_queue[d_ind, f_ind]
        # mu_, sig_ = mu_.detach() / k_dom, sig_.detach() / k_dom
        # #### equal ratio of mu, sig formation (just one mu, sig)

        # # substitute (like AdaIN)
        # return x_normed*sig_ + mu_
    
        #### mixstyle like
        mu_mix = mu*lmda + mu1 * (1-lmda)
        sig_mix = sig*lmda + sig1 * (1-lmda)
        # mu_mix = mu2*lmda + mu1 * (1-lmda)
        # sig_mix = sig2*lmda + sig1 * (1-lmda)
        return x_normed*sig_mix + mu_mix