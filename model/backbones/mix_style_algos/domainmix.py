import torch
import torch.nn as nn
import random
from torch.distributions.normal import Normal

class DomainMix(nn.Module):
    def __init__(self, num_features, num_domains, momentum=0.9, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.momentum = momentum
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

        self.register_buffer('mean', torch.zeros(num_domains, self.num_features))
        self.register_buffer('var', torch.zeros(num_domains, self.num_features))
        self.num_batch = 0

    def forward(self, input, domain=None):
        if not self.training:
            return input
        
        moment = self.momentum
        eps = self.eps

        x = input.detach()
        #### ema update
        for i in range(self.num_domains):
            if not i in domain:
                continue
            mean = x[domain==i].mean([0,1]).detach()
            var = x[domain==i].var([0,1]).detach()
            if self.num_batch == 0:
                self.mean[i] = mean
                self.var[i] = var
            else:
                mean_old = self.mean[i]
                var_old = self.var[i]
                self.mean[i] = moment*mean_old + (1-moment)*mean
                self.var[i] = moment*var_old + (1-moment)*var
            if torch.isnan(self.mean).any() or torch.isnan(self.var).any():
                assert False, "NaN"
        self.num_batch = self.num_batch + 1

        if random.random() > self.p:
            return input
        
        B = x.size(0)
        x_normed = (x - x.mean(1, keepdim=True)) / torch.sqrt(x.var(1, keepdim=True) + eps)

        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)

        domain_select = torch.randint_like(domain, self.num_domains)

        # Distris = Normal(self.mean[domain_select], self.var[domain_select] + self.eps)
        # samples = Distris.sample()

        #### mixup type
        # x_mix = lmda * x + (1-lmda) * (x_normed + )
        # x_mix = lmda * x + (1-lmda) * samples

        # #### mixup like (mix x)
        # mean_new, var_new = self.mean[domain_select].unsqueeze(1), self.var[domain_select].unsqueeze(1)
        # x_new = x_normed * torch.sqrt(var_new + eps) + mean_new
        # x_mix = lmda * x + (1 - lmda) * x_new
        # return x_mix
    
        # #### multi mix (to do)
        # dom_num = random.randint(1, self.num_domains)
        # dom_ind = random.choices(range(0, self.num_domains), k=dom_num)

        #### mixstyle (mix mean and sigma)
        mean_mix = lmda * x.mean(1,keepdim=True) + (1-lmda) * self.mean[domain_select].detach().unsqueeze(1)
        var_mix = lmda * x.var(1, keepdim=True) + (1-lmda) * self.var[domain_select].detach().unsqueeze(1)
        x_mix = x_normed*torch.sqrt(var_mix + eps) + mean_mix
        return x_mix