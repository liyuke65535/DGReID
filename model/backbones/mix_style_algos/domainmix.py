import torch
import torch.nn as nn
import random
from torch.distributions.normal import Normal

from loss.triplet_loss import euclidean_dist, hard_example_mining

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
        self.mean.requires_grad = False
        self.var.requires_grad = False

        self.loss = nn.SoftMarginLoss()

    def forward(self, input, labels, domain=None):
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
        self.num_batch = self.num_batch + 1

        if random.random() > self.p:
            return input
        
        B = x.size(0)
        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)
        #### make sure that inds are all different from domain
        d_ind = random.choices(range(1, self.num_domains), k=B)
        d_ind = torch.tensor(d_ind, device=domain.device) + domain
        domain_select = d_ind % self.num_domains

        # ############ efdmix like #############

        # Distris = Normal(self.mean[domain_select], torch.sqrt(self.var[domain_select]+eps))
        # samples = Distris.sample([129]).permute(1,0,2)

        # value_x, index_x = torch.sort(x, dim=0)
        # inverse_index = index_x.argsort(-1)
        # samples_sorted = torch.sort(samples, dim=1)[0]
        # x_new = samples_sorted.gather(-1, inverse_index) * (1-lmda)
        # # x_mix = x + (x_new - x.detach() * (1-lmda))
        # x_mix = x * lmda + x_new * (1-lmda)
        # return x_mix
        # ############ efdmix like #############

        ############ mixstyle like ###########
        x_normed = (x - x.mean(1, keepdim=True)) / torch.sqrt(x.var(1, keepdim=True) + eps)

        #### mixup type
        # x_mix = lmda * x + (1-lmda) * (x_normed + )
        # x_mix = lmda * x + (1-lmda) * samples

        #### mixup like (mix x)  same with mixstyle like
        mean_new, var_new = self.mean[domain_select].unsqueeze(1), self.var[domain_select].unsqueeze(1)
        x_new = x_normed * torch.sqrt(var_new + eps) + mean_new
        x_mix = lmda * x + (1 - lmda) * x_new

        ############# expand hard samples #############
        #### hard negetive
        hg = Normal(self.mean[domain], torch.sqrt(self.var[domain]+eps)).sample([129]).permute(1,0,2)
        #### hard positive
        hp = x_mix #### or x_new
        x_expand = torch.cat([x.reshape(4,B/4,129,768), hp.reshape(4,B/4,129,768)],dim=0).reshape(B,129,768)
        labels_new = torch.cat([labels.reshape(4,B/4),labels.reshape(4,B/4)], dim=0).reshape(B)
        x_expand = torch.cat([x_expand, hg], dim=0)
        labels_new = torch.cat([labels_new, -torch.ones([B])], dim=0)
        dist_mat = euclidean_dist(x_expand, x_expand)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels_new)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        tri_hard_loss = self.loss(dist_an - dist_ap, y)
        ############# expand hard samples #############
        return x_mix
    
        # #### multi mix (to do)
        # dom_num = random.randint(1, self.num_domains)
        # dom_ind = random.choices(range(0, self.num_domains), k=dom_num)

        # #### mixstyle like (mix mean and sigma)
        # mean_mix = lmda * x.mean(1,keepdim=True) + (1-lmda) * self.mean[domain_select].detach().unsqueeze(1)
        # var_mix = lmda * x.var(1, keepdim=True) + (1-lmda) * self.var[domain_select].detach().unsqueeze(1)
        # x_mix = x_normed*torch.sqrt(var_mix + eps) + mean_mix
        # return x_mix
        ############ mixstyle like ###########