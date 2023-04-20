import random
import torch
import torch.nn as nn

from model.backbones.vit_pytorch import trunc_normal_


class DomainQueue(nn.Module):

    def __init__(self, num_features, num_domains, p=0.5, alpha=0.1, eps=1e-6, mix='diff_domain', capacity=1024):
        """
        Args:
          p (float): probability of using mix.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix. (random / diff_domain)
        """
        super().__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        
        self.sum = list(0 for _ in range(num_domains + 1))
        self.capacity = capacity
        self.register_buffer('mean_queue', torch.zeros(num_domains + 1, capacity, num_features))
        self.register_buffer('sig_queue', torch.ones(num_domains + 1, capacity, num_features))
        self.mean_queue.requires_grad = False
        self.sig_queue.requires_grad = False

        trunc_normal_(self.mean_queue)
        trunc_normal_(self.sig_queue)

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

        if self.mix == 'random':
            d_ind1 = torch.randint_like(domain, 0, self.num_domains+1)
        #### make sure that inds are all different from domain
        elif self.mix == 'diff_domain':
            d_ind1 = torch.zeros_like(domain)
            for i in range(B):
                # lst = list(range(0, self.num_domains)) # no sync
                lst = list(range(0, self.num_domains+1)) # sync
                lst.remove(domain[i])
                d_ind1[i] = random.choice(lst)
        else:
            assert False, "not implemented mix way: {}".format(self.mix)
        f_ind1 = torch.tensor([random.randint(0, self.sum[d_ind1[i]] % self.capacity) for i in range(B)])
        # f_ind2 = torch.tensor([random.randint(0, self.sum[d_ind1[i]] % self.capacity) for i in range(B)])
        mu1 = self.mean_queue[d_ind1, f_ind1].unsqueeze(1)
        sig1 = self.sig_queue[d_ind1, f_ind1].unsqueeze(1)
        # mu2 = self.mean_queue[d_ind2, f_ind2].unsqueeze(1)
        # sig2 = self.sig_queue[d_ind2, f_ind2].unsqueeze(1)

        # #### part into groups, each group adopts the same mix operation (single domain in batches only)
        # mix_num = 16
        # repeat_times = B // mix_num
        # lmda = self.beta.sample((mix_num,1,1)).repeat(repeat_times,1,1).to(x.device)
        # d_ind1 = random.choices(range(1, self.num_domains), k=mix_num)
        # d_ind1 = torch.tensor(d_ind1, device=domain.device)
        # # d_ind1 = torch.tensor(d_ind1, device=domain.device) + torch.unique(domain)
        # # d_ind1 = d_ind1 % self.num_domains
        # d_ind1 = d_ind1.unsqueeze(0).repeat(repeat_times,1).view(-1)
        # f_ind1 = torch.tensor([random.randint(0, self.sum[d_ind1[i]] % self.capacity) for i in range(mix_num)])
        # f_ind1 = f_ind1.unsqueeze(0).repeat(repeat_times,1).view(-1)
        # mu1 = self.mean_queue[d_ind1, f_ind1].unsqueeze(1)
        # sig1 = self.sig_queue[d_ind1, f_ind1].unsqueeze(1)

        # #### fuse all domains
        # idxs = torch.tensor([[random.randint(0, self.sum[i] % self.capacity) for _ in range(B)]for i in range(self.num_domains)]).to(x.device)
        # ratios = torch.rand([self.num_domains,B])
        # ratios = ratios.softmax(dim=0).to(x.device)
        # mu1 = torch.zeros([B,self.num_features], dtype=x.dtype).to(x.device)
        # sig1 = torch.zeros([B,self.num_features], dtype=x.dtype).to(x.device)
        # for i in range(self.num_domains):
        #     mu1 = mu1 + self.mean_queue[i].index_select(0, idxs[i]) * ratios[i].unsqueeze(1)
        #     sig1 = sig1 + self.sig_queue[i].index_select(0, idxs[i]) * ratios[i].unsqueeze(1)
        # mu1 = mu1.unsqueeze(1) / self.num_domains
        # sig1 = sig1.unsqueeze(1) / self.num_domains


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

        #### novel style enqueue
        sum = self.sum[-1] % self.capacity
        rest = self.capacity - sum
        if B > rest:
            self.mean_queue[-1, sum:] = mu_mix.squeeze()[:rest]
            self.mean_queue[-1, :B-rest] = mu_mix.squeeze()[rest:]
            self.sig_queue[-1, sum:] = sig_mix.squeeze()[:rest]
            self.sig_queue[-1, :B-rest] = sig_mix.squeeze()[rest:]
        else:
            self.mean_queue[-1, sum:sum+B] = mu_mix.squeeze()
            self.sig_queue[-1, sum:sum+B] = sig_mix.squeeze()
        self.sum[-1] = self.sum[-1] + int(B)

        # mu_mix = mu2*lmda + mu1 * (1-lmda)
        # sig_mix = sig2*lmda + sig1 * (1-lmda)
        return x_normed*sig_mix + mu_mix


class DomainQueue_2d(nn.Module):
    def __init__(self, num_features, num_domains, p=0.5, alpha=0.1, eps=1e-6, mix='diff_domain', capacity=1024):
        super().__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        
        self.sum = list(0 for _ in range(num_domains + 1))
        self.capacity = capacity
        self.register_buffer('mean_queue', torch.zeros(num_domains + 1, capacity, num_features))
        self.register_buffer('sig_queue', torch.ones(num_domains + 1, capacity, num_features))
        self.mean_queue.requires_grad = False
        self.sig_queue.requires_grad = False

        trunc_normal_(self.mean_queue)
        trunc_normal_(self.sig_queue)

    def forward(self, x, domain=None):
        if not self.training:
            return x

        for i in range(self.num_domains):
            if not i in domain:
                continue
            mean = x[domain==i].mean([2,3]).detach()
            sig = (x[domain==i].var([2,3])+self.eps).sqrt().detach()
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
        mu = x.mean(dim=[2,3], keepdim=True)
        var = x.var(dim=[2,3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            d_ind1 = torch.randint_like(domain, 0, self.num_domains+1)
        #### make sure that inds are all different from domain
        elif self.mix == 'diff_domain':
            d_ind1 = torch.zeros_like(domain)
            for i in range(B):
                # lst = list(range(0, self.num_domains)) # no sync
                lst = list(range(0, self.num_domains+1)) # sync
                lst.remove(domain[i])
                d_ind1[i] = random.choice(lst)
        else:
            assert False, "not implemented mix way: {}".format(self.mix)
        f_ind1 = torch.tensor([random.randint(0, self.sum[d_ind1[i]] % self.capacity) for i in range(B)])
        mu1 = self.mean_queue[d_ind1, f_ind1][:, :, None, None]
        sig1 = self.sig_queue[d_ind1, f_ind1][:, :, None, None]
    
        #### mixstyle like
        mu_mix = mu*lmda + mu1 * (1-lmda)
        sig_mix = sig*lmda + sig1 * (1-lmda)

        #### novel style enqueue
        sum = self.sum[-1] % self.capacity
        rest = self.capacity - sum
        if B > rest:
            self.mean_queue[-1, sum:] = mu_mix.squeeze()[:rest]
            self.mean_queue[-1, :B-rest] = mu_mix.squeeze()[rest:]
            self.sig_queue[-1, sum:] = sig_mix.squeeze()[:rest]
            self.sig_queue[-1, :B-rest] = sig_mix.squeeze()[rest:]
        else:
            self.mean_queue[-1, sum:sum+B] = mu_mix.squeeze()
            self.sig_queue[-1, sum:sum+B] = sig_mix.squeeze()
        self.sum[-1] = self.sum[-1] + int(B)

        return x_normed*sig_mix + mu_mix