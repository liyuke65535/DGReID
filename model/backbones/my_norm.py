'''
@File    :   my_norm.py
@Time    :   2023/02/07 19:44:23
@Author  :   liyuke 
'''
import torch
import torch.nn as nn

# ##### not working version 1
# class LBN(nn.Module):
#     def __init__(self, seq_len: int, eps: float = 0.00001, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, device='cuda') -> None:
#         super().__init__()
#         self.bn = nn.BatchNorm1d(num_features=seq_len, eps=eps, momentum = momentum, affine = affine, track_running_stats=track_running_stats, device=device)
#         self.bn.reset_parameters()
#         self.weight = self.bn.weight
#         self.bias = self.bn.bias
        
#     def forward(self, x):
#         # x = x.permute(0,2,1) # B, N, C -> B, C, N
#         x = self.bn(x)
#         # x = x.permute(0,2,1)
#         return x
   
### version 2
class LBN(nn.Module):
    def __init__(self, num_features, num_patches, eps=1e-06, affine=True, track_running_stats=True, momentum=None, device=None, dtype=None) -> None:
        super().__init__()
        self.eps = eps
        self.track_running_stats = track_running_stats
        if affine:
            self.register_buffer("weight", torch.ones(num_features))
            self.register_buffer("bias", torch.zeros(num_features))
            self.reset_parameters()
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_patches))
            self.register_buffer("running_var", torch.ones(num_patches))
            # self.running_mean=torch.ones(129, device=device, dtype=dtype)
            # self.running_var=torch.ones(129, device=device, dtype=dtype)
            self.momentum = momentum
            self.count = 0
            self.reset_running_stats()
        
    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.1)
        nn.init.constant_(self.bias, 0.0)
        
    def reset_running_stats(self):
        nn.init.constant_(self.running_mean, 0.0)
        nn.init.constant_(self.running_var, 0.0)

    def forward(self, x):
        B,N,C = x.shape
        eps = self.eps
        if not self.training:
            r_m = self.running_mean.unsqueeze(0).unsqueeze(-1)
            r_v = self.running_var.unsqueeze(0).unsqueeze(-1)
            res = (x - r_m) / torch.sqrt(r_v + eps)
            return self.weight * res + self.bias

        mean = x.mean(dim=(0,2), keepdim=True)
        var = x.var(dim=(0,2), keepdim=True)
        res = (x - mean) / torch.sqrt(var + eps)
        res = self.weight * res + self.bias
        
        # no statics
        if not self.track_running_stats:
            return res

        # MA
        if self.momentum is None:
            if self.count == 0:
                self.running_mean = mean.squeeze()
                self.running_var = var.squeeze()
            else:
                self.running_mean = (self.running_mean * self.count + mean.squeeze())\
                    / (self.count + 1)
                self.running_var = (self.running_var * self.count + var.squeeze())\
                    / (self.count + 1)
            self.count = self.count + 1
            return res
        
        # EMA
        mo = self.momentum
        self.running_mean = (1-mo) * self.running_mean + mo * mean.squeeze()
        self.running_var = (1-mo) * self.running_var + mo * var.squeeze()
        
        return res
    
if __name__ == '__main__':
    # x = torch.rand([64, 129, 768]).to('cuda')
    # norm = LBN(seq_len=129)
    # out = norm(x)
    # print(x)
    # print(out)
    x = torch.randn([4,8,4,4])
    norm = nn.BatchNorm2d(8)
    print(norm.weight.shape)
    out = norm(x)
    print(x)
    print(out)