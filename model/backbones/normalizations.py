'''
@File    :   normalizations.py
@Time    :   2023/02/07 19:44:04
@Author  :   liyuke 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
# nn.BatchNorm1d().forward
class BatchNorm(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0, bias_init=0.0, track_running_stats=True):
        super().__init__(num_features, eps=eps, momentum=momentum, track_running_stats=track_running_stats)
        if weight_init is not None: self.weight.data.fill_(weight_init)
        if bias_init is not None: self.bias.data.fill_(bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)
        
    def forward(self, x):
        self._check_input_dim(x)
        B, N, C = x.size()
        assert C == self.num_features
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return F.batch_norm(
            x.permute(0,2,1),
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        ).permute(0,2,1)

class InstanceNorm(nn.InstanceNorm1d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        if input.dim() == self._get_no_batch_dim():
            return self._handle_no_batch_input(input)

        return F.instance_norm(
            input.permute(0,2,1),
            self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps).permute(0,2,1)

"""
LayerNorm with tracking running stats.
!!! CANNOT WORKING !!!
because of the unique normalize dim of LN,
which leads to different batch size
of training and testing.
"""
class LBN(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-06, elementwise_affine=True, track_running_stats=True, momentum=0.1, device=None, dtype=None) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.running_mean=torch.ones(129, device=device, dtype=dtype)
            self.running_var=torch.ones(129, device=device, dtype=dtype)
            self.momentum = momentum

    def forward(self, input):
        if not self.training:
            res = (input - self.running_mean) / torch.sqrt(self.running_var + eps)
            return self.weight * res + self.bias

        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, keepdim=True)
        eps = self.eps
        if not self.track_running_stats:
            res = (input - mean) / torch.sqrt(var + eps)
            return self.weight * res + self.bias

        momentum = self.momentum
        running_mean = self.running_mean.to(input.device.type)
        running_var = self.running_var.to(input.device.type)

        running_mean = (1-momentum) * running_mean + momentum * mean
        running_var = (1-momentum) * running_var + momentum * var
        
        x_normalized = (input-running_mean)/torch.sqrt(running_var+eps)

        # update running stats
        self.running_mean = running_mean.detach()
        self.running_var = running_var.detach()
        return self.weight * x_normalized + self.bias # scale