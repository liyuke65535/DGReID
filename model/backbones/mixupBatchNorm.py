import torch
import torch.nn as nn
import numpy as np

class MixUpBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats)

        self.register_buffer('mean1', torch.zeros(self.num_features))
        self.register_buffer('var1', torch.zeros(self.num_features))
        self.register_buffer('mean2', torch.zeros(self.num_features))
        self.register_buffer('var2', torch.zeros(self.num_features))
        self.device_count = torch.cuda.device_count()

    def forward(self, input, MTE='', save_index=0):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            if MTE == 'sample':
                from torch.distributions.normal import Normal
                Distri1 = Normal(self.mean1, self.meta_var1+1e-12)
                Distri2 = Normal(self.mean2, self.meta_var2+1e-12)
                sample1 = Distri1.sample([input.size(0), ])
                sample2 = Distri2.sample([input.size(0), ])
                lam = np.random.beta(1., 1.)
                inputmix1 = lam * sample1 + (1-lam) * input
                inputmix2 = lam * sample2 + (1-lam) * input

                mean1 = inputmix1.mean(dim=0)
                var1 = inputmix1.var(dim=0, unbiased=False)
                mean2 = inputmix2.mean(dim=0)
                var2 = inputmix2.var(dim=0, unbiased=False)

                output1 = (inputmix1 - mean1[None, :]) / (torch.sqrt(var1[None, :] + self.eps))
                output2 = (inputmix2 - mean2[None, :]) / (torch.sqrt(var2[None, :] + self.eps))
                if self.affine:
                    output1 = output1 * self.weight[None, :] + self.bias[None, :]
                    output2 = output2 * self.weight[None, :] + self.bias[None, :]
                return [output1, output2]

            else:
                mean = input.mean(dim=0)
                # use biased var in train
                var = input.var(dim=0, unbiased=False)
                n = input.numel() / input.size(1)

                with torch.no_grad():
                    running_mean = exponential_average_factor * mean \
                                   + (1 - exponential_average_factor) * self.running_mean
                    # update running_var with unbiased var
                    running_var = exponential_average_factor * var * n / (n - 1) \
                                  + (1 - exponential_average_factor) * self.running_var
                    self.running_mean.copy_(running_mean)
                    self.running_var.copy_(running_var)
                    if save_index == 1:
                        self.mean1.copy_(mean)
                        self.var1.copy_(var)
                    elif save_index == 2:
                        self.mean2.copy_(mean)
                        self.var2.copy_(var)

        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :]) / (torch.sqrt(var[None, :] + self.eps))
        if self.affine:
            input = input * self.weight[None, :] + self.bias[None, :]

        return input