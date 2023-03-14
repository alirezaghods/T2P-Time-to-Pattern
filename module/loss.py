import torch
from torch import nn, Tensor

recon_function = nn.MSELoss()


def kld_function(alpha:Tensor, a, lambda_1, lambda_2, device) -> Tensor:
    """
    KLD function for concrete distribution
    """
    eps = 1e-7
    S = 1000
    N = alpha.size(1) * alpha.size(2)
    dtype = torch.float
    alpha_expanded = alpha[0,:,:,:,0].expand(alpha[0,:,:,:,0].size(0),alpha[0,:,:,:,0].size(1),S)
    one = torch.ones(alpha[0,:,:,:,0].size(),dtype=dtype, device=device, requires_grad=False)
    U = torch.rand(alpha_expanded.size(),dtype=dtype,device=device,requires_grad=False) 
    fixed = torch.log(one.mul(a * lambda_2 / lambda_1)).add_(2.) 
    log_alpha = torch.log(alpha[0,:,:,:,0]+eps).mul(-lambda_2 / lambda_1)
    integral = torch.sum(torch.log((((U).div(1-U+eps)).mul(alpha_expanded)).add(eps).pow(-lambda_2/lambda_1).mul(a).add(1.)),dim=2,keepdim=True).div(S).mul(-2.)
    kld = torch.sum(fixed.mul_(-1).add_(log_alpha.mul_(-1)).add_(integral.mul_(-1))).div_(N)
    return kld

def loss(output, input, alpha, a, lambda_1, lambda_2, device):
    return  recon_function(output,input) + kld_function(alpha, a, lambda_1, lambda_2, device)
