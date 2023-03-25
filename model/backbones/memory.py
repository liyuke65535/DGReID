import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd
import numpy as np

class MC(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            # grad_inputs = grad_outputs.mm(ctx.features)
            grad_inputs = grad_outputs.mm(torch.as_tensor(ctx.features, dtype=grad_outputs.dtype))

        return grad_inputs, None, None, None


def mc(inputs, indexes, features, momentum=0.5):
    return MC.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class MemoryClassifier(nn.Module):
    def __init__(self, num_features, num_classes, temp=1., momentum=0.2):
        super(MemoryClassifier, self).__init__()
        self.num_features = num_features
        self.num_samples = num_classes
        self.momentum = momentum
        self.temp = temp #### temperature

        self.register_buffer('features', torch.zeros(num_classes, num_features))
        self.register_buffer('labels', torch.zeros(num_classes).long())

    def MomentumUpdate(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] = self.features[y] / self.features[y].norm()


    def forward(self, inputs, indexes):

        sim = mc(inputs, indexes, self.features, self.momentum) ## B * C

        sim = sim / self.temp

        #### 和论文里面不太一样
        loss = F.cross_entropy(sim, indexes) # indexes: smoothed labels
        return loss
        
        # #### 按原文来的loss (报nan)
        # idx = indexes.max(1)[1]
        # numerator = torch.exp((inputs * self.features[idx]).sum(1))
        # denominator = torch.exp(inputs @ self.features.t()).sum(1)
        # loss = -torch.log(numerator / denominator)
        # return loss.sum()

from loss.triplet_loss import euclidean_dist
class FeatureMemory(nn.Module):
    def __init__(self, num_features, num_pids, momentum=0.9) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.momentum = momentum

        self.avai_pids = []
        self.register_buffer("feats", torch.zeros(num_pids, num_features))
        self.feats.requires_grad = False

        self.ranking_loss = nn.SoftMarginLoss()

        self.saved_tensors = None
    def momentum_update(self):
        if self.saved_tensors is None:
            print("None saved tensors!!!!!!")
            return
        x, labels = self.saved_tensors[0], self.saved_tensors[1]
        self.feats[labels] = self.feats[labels] * self.momentum + x * (1-self.momentum)

    def save_tensors(self, x, labels):
        self.saved_tensors = [x.detach(), labels]

    def forward(self, x, labels):
        dist_mat = euclidean_dist(x,x)
        assert dist_mat.dim() == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)
        is_pos = labels.expand(N, N).eq(labels.expand(N,N).t())
        # dist_ap, relative_p_inds = torch.max(
        #     dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
        fea = self.feats
        dist_mat2 = euclidean_dist(x, fea)
        one_hot_labels = torch.zeros([N, self.num_pids]).scatter_(1, labels.unsqueeze(1).data.cpu(), 1)
        dist_ap, relative_p_inds = torch.max(
            dist_mat2[one_hot_labels.bool()].contiguous().view(N, -1), 1, keepdim=True)
        dist_an, relative_n_inds = torch.min(
            dist_mat2[~one_hot_labels.bool()].contiguous().view(N, -1), 1, keepdim=True)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an - dist_ap, y)

        self.save_tensors(x, labels)
        return loss