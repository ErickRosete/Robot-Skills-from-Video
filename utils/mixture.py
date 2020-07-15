#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 17:23:58 2020

@author: suresh, eric, jessica
"""

import numpy as np
import torch

# =============================================================================
# references : 
# 1. https://github.com/Rayhane-mamah/Tacotron-2/blob/d13dbba16f0a434843916b5a8647a42fe34544f5/wavenet_vocoder/models/mixture.py#L90-L93
# 2. https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py#L46
# 3. https://github.com/Rayhane-mamah/Tacotron-2/issues/155
# 4. https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/mixture.py [pytorch]
# =============================================================================

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    m = torch.max(x, dim=-1, keepdims=True)
    return x - m - torch.log(torch.sum(torch.exp(x-m), dim=-1, keepdims=True))
  
def discretized_mix_logistic_loss(x, l, n_mix = 10, num_classes=256, log_scale_min=-7.0, reduce=True):
    """
    Parameters
    ----------
    l : TYPE
        DESCRIPTION. predicted output (B x C x T)
    x : TYPE
        DESCRIPTION. target (B x T x 1)
    num_classes : TYPE, optional
        DESCRIPTION. The default is 256.
    log_scale_min : TYPE, optional
        DESCRIPTION. The default is -7.0.

    Returns
    -------
    TYPE
        DESCRIPTION. Tensor loss

    """
    
    # (B x C x T) -> (B x T x C)
    l = l.permute(0, 2, 1)
    
    # [batch_size, T, num_mixtures] x 3
    logit_probs = l[:, :, :n_mix]
    means = l[:, :, n_mix:2 * n_mix]
    log_scales = torch.clamp(l[:, :, 2* n_mix: 3 * n_mix], min=log_scale_min)
    
    # B x T x 1 -> B x T x num_mixtures
    x = x.expand_as(means)
    
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)
    
    log_cdf_plus = plus_in - torch.nn.functional.softplus(plus_in) # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -torch.nn.functional.softplus(min_in) # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min # probability for all other cases
    
    mid_in = inv_stdv * centered_x

    log_pdf_mid = mid_in - log_scales - 2. * torch.nn.functional.softplus(mid_in) #log probability in the center of the bin, to be used in extreme cases (not actually used in this code)
    
    # log_probs = torch.where(x < -0.999, log_cdf_plus,
    #                      torch.where(x > 0.999, log_one_minus_cdf_min,
    #                               torch.where(cdf_delta > 1e-5,
    #                                        torch.log(torch.max(cdf_delta, torch.tensor(1e-12))),
    #                                        log_pdf_mid - torch.log(torch.tensor((num_classes - 1) / 2)))
    #                               )
    #                      )
    
    
    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * \
        torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
        (1. - inner_inner_cond) * (log_pdf_mid - np.log((num_classes - 1) / 2))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    
    log_probs = log_probs + torch.nn.functional.log_softmax(logit_probs, dim=-1)
    
    if reduce:
        return -torch.sum(log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs).unsqueeze(-1)

def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot

def sample_from_discretized_mix_logistic(y, n_mix = 10, log_scale_min=-7., clamp_log_scale=False):
    """
    Parameters
    ----------
    y : TYPE
        DESCRIPTION. (B x C x T)
    log_scale_min : TYPE, optional
        DESCRIPTION. The default is -7..

    Returns
    -------
    TYPE
        DESCRIPTION. Tensor: sample in range of [-1, 1]

    """
    # (B x C x T) -> (B x T x C)
    y = y.permute(0, 2, 1)
    
    logit_probs = y[:, :, :n_mix]
    means = y[:, :, n_mix:2 * n_mix]
    scales = torch.sum(y[:, :, 2 * n_mix:3 * n_mix])
    
    #sample mixture indicator from softmax
    temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=-1)
    
    #[batch_size, ...] -> [batch_size, ..., n_mix]
    #one_hot = torch.nn.functional.one_hot(argmax, num_classes=n_mix)
    one_hot = to_one_hot(argmax, n_mix)
    #select logistic parameters
    means = torch.sum(means * one_hot, dim=-1)
    log_scales = torch.sum(scales * one_hot, dim=-1)
    if clamp_log_scale:
        log_scales = torch.clamp(log_scales, min=log_scale_min)
    
    #sample from logistic & clip to interval
	#we don't actually round to the nearest 8-bit value when sampling
    u = means.data.new(means.size()).uniform_(1e-5, 1.0 - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1 - u))
    
    x = torch.clamp(x, min=-1., max=1.)
    return x

if __name__ == '__main__':
    BATCH_SIZE = 32
    N_DOF_ROBOT = 9
    N_MIX = 10

    y = torch.rand(BATCH_SIZE, 3 * N_MIX, N_DOF_ROBOT)#.cuda()
    sample = sample_from_discretized_mix_logistic(y)
    print(sample)
    
    prediction = torch.rand(BATCH_SIZE, 3 * N_MIX, N_DOF_ROBOT)#.cuda()
    target = torch.rand(BATCH_SIZE, N_DOF_ROBOT, 1)#.cuda()
    loss = discretized_mix_logistic_loss(target, prediction, num_classes=256, log_scale_min=-7.0)
    print(loss)
    
