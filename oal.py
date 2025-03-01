import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

EPS = 1e-9

# Orientation Anisotropic Loss (OAL)
def orientation_anisotropic_loss(kernels: torch.Tensor, lambda_val: float = 8, eta: float = -0.5) -> torch.Tensor:
    """
    Computes the Orientation Anisotropic Loss (OAL) for a set of convolution kernels.

    Parameters:
        kernels (torch.Tensor): A tensor of shape (K, C, H, W) representing the convolution kernels from the APC layer.
        lambda_val (float): Weighting coefficient for variance modulation (default 8).
        eta (float): Lower bound for the standard deviation term (default -0.5).

    Returns:
        torch.Tensor: A scalar tensor representing the computed OAL loss.
    """
    K, C, H, W = kernels.shape
    oal_loss = 0.0

    for i in range(K):
        j = (i + K // 2) % K
        k_i, k_j = kernels[i], kernels[j]

        # Cross-convolution term
        cross_norm = torch.norm(k_i * k_j, p=2)

        # Self-convolution term with 180-degree rotation
        crt_k_i = torch.flip(k_i, dims=[-2, -1])
        self_conv_norm = torch.norm(k_i * crt_k_i, p=2)

        # Standard deviation term
        std_val = torch.std(k_i)

        # Loss term calculation
        term = cross_norm / (cross_norm + self_conv_norm + lambda_val * max(eta, -std_val) + EPS)
        oal_loss += term

    return oal_loss / K

# Combined loss function
def get_loss(P_out, P_aux, Y, kernels, lambda_seg=1.0, lambda_aux=0.5, lambda_oal=0.1):
    """
    Computes the overall loss as defined in the WS-DMF paper.
    """
    focal_loss = FocalLoss()(P_out, Y)
    dice_loss = DiceLoss()(P_aux, Y)
    oal_loss = orientation_anisotropic_loss(kernels)

    total_loss = lambda_seg * focal_loss + lambda_aux * dice_loss + lambda_oal * oal_loss
    return total_loss, focal_loss.item(), dice_loss.item(), oal_loss.item()

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-6):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, pr, gt):
        pr = torch.clamp(pr, self.eps, 1 - self.eps)
        loss1 = -gt * torch.log(pr) * torch.pow(1 - pr, self.gamma)
        loss2 = -(1 - gt) * torch.log(1 - pr) * torch.pow(pr, self.gamma)
        return (loss1 + loss2).mean()

# Dice Loss
class DiceLoss(nn.Module):
    def forward(self, pr, gt, smooth=1.0):
        pr, gt = pr.view(-1), gt.view(-1)
        inter = (pr * gt).sum()
        union = (pr + gt).sum()
        return 2 - (2 * inter + smooth) / (union + smooth)

# RAdamW Optimizer
class RAdamW(Optimizer):
    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RAdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdamW does not support sparse gradients')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'])
        return loss

# ReduceLR Scheduler
def get_scheduler(optimizer):
    return ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2, 
                              cooldown=2, min_lr=1e-5, eps=1e-9, verbose=True)
