# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn


class RealNVP(nn.Module):
    """
        RealNVP: flow-based generative model
                 "Density estimation using Real NVP"
        Args:
            get_scale_net (function): return layers to build scale net
            get_trans_net (function): return layers to build transition net
            mask (torch.Tensor): binary mask for decoupling
            prior (torch.Distribution): π(z)
    """

    def __init__(self, get_scale_net, get_trans_net, mask, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.mask = mask
        self.s = torch.nn.ModuleList(
            [get_scale_net() for _ in range(len(mask))])
        self.t = torch.nn.ModuleList(
            [get_trans_net() for _ in range(len(mask))])
        self._init()

    def _init(self):
        for m in self.t:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)
        for m in self.s:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)

    def backward_p(self, x):
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])  # torch.exp(s): betas
            t = self.t[i](z_) * (1 - self.mask[i])  # gammas
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det
