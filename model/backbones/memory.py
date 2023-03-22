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
