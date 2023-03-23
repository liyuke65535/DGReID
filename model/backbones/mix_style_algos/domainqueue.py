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

    def forward(self, x, domain=None):
        if not self.training:
            return x

        for i in range(self.num_domains):
            if not i in domain:
                continue
            mean = x[domain==i].mean(1)
            sig = (x[domain==i].var(1)+self.eps).sqrt()
            length = (domain==i).sum()

            sum = self.sum[i] % self.capacity
            rest = self.capacity - sum
            if length > rest:
                self.mean_queue[i, sum:] = mean[:rest].detach()
                self.mean_queue[i, :length-rest] = mean[rest:].detach()
                self.sig_queue[i, sum:] = sig[:rest].detach()
                self.sig_queue[i, :length-rest] = sig[rest:].detach()
            # elif sum == 0:
            #     self.mean_queue[i] = mean.expand_as(self.mean_queue[i]).detach()
            #     self.sig_queue[i] = sig.expand_as(self.sig_queue[i]).detach()
            else:
                self.mean_queue[i, sum:sum+length] = mean.detach()
                self.sig_queue[i, sum:sum+length] = sig.detach()
            self.sum[i] = self.sum[i] + int(length)

            # for ind in range(length):
            #     sum = self.sum[i] % self.capacity
            #     self.mean_queue[i, sum] = mean[ind].detach()
            #     self.sig_queue[i, sum] = sig[ind].detach()
            #     self.sum[i] = self.sum[i] + 1

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

        #### random indexes (1 ~ num domains - 1)
        #### make sure that inds are all different from domain
        d_inds = random.choices(range(1, self.num_domains), k=B)
        d_inds = torch.tensor(d_inds, device=domain.device) + domain
        d_inds = d_inds % self.num_domains
        f_inds = torch.randint_like(domain, self.capacity)
        mu_ = self.mean_queue[d_inds, f_inds].unsqueeze(1)
        sig_ = self.sig_queue[d_inds, f_inds].unsqueeze(1)

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
        mu_mix = mu*lmda + mu_ * (1-lmda)
        sig_mix = sig*lmda + sig_ * (1-lmda)
        return x_normed*sig_mix + mu_mix