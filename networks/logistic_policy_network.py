import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils.constants as constants
import numpy as np

#Smooth approximation to the maximum
def log_sum_exp(x):
	m, _ = torch.max(x, -1)
	m2, _ = torch.max(x, -1, keepdims=True)
	return m + torch.log(torch.sum(torch.exp(x-m2), -1))

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

#Selects which logistic distribution to sample from
def gumbel_softmax(logits, temperature): 
    y = gumbel_softmax_sample(logits, temperature) #logits [*, n_class]
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y    #[*, n_class] an one-hot vector

class LogisticPolicyNetwork(nn.Module):
    def __init__(self, n_dist=5):
        super(LogisticPolicyNetwork, self).__init__()
        self.temp = 1
        self.in_features = (constants.VISUAL_FEATURES + constants.N_DOF_ROBOT) + \
                            constants.VISUAL_FEATURES + constants.PLAN_FEATURES
        hidden_size = 2048
        self.out_features = 9
        self.n_dist = n_dist
        self.rnn = nn.RNN(input_size=self.in_features, hidden_size=hidden_size, num_layers=2, \
                        nonlinearity='relu', bidirectional=False, batch_first=True)
        self.mean_fc = nn.Linear(hidden_size, self.out_features * self.n_dist)
        self.log_scale_fc = nn.Linear(hidden_size, self.out_features * self.n_dist)
        self.prob_fc = nn.Linear(hidden_size, self.out_features * self.n_dist)

    def loss(self, probs, log_scales, means, actions, log_scale_min=-7.0, num_classes=256):
        #Appropiate scale
        log_scales = torch.clamp(log_scales, min=log_scale_min)
        #Transform to logits 
        logit_probs = F.softmax(probs, dim=-1)
        #Brodcast actions (B, A, N_DIST)
        actions = actions.unsqueeze(-1).expand_as(means)

        #Approximation of CDF derivative (PDF)
        centered_actions = actions - means 
        inv_stdv = torch.exp(-log_scales) 
        plus_in = inv_stdv * (centered_actions + 1. / (num_classes - 1)) 
        cdf_plus = torch.sigmoid(plus_in) 
        min_in = inv_stdv * (centered_actions - 1. / (num_classes - 1)) 
        cdf_min = torch.sigmoid(min_in) 

        #Corner Cases
        log_cdf_plus = plus_in -F.softplus(plus_in) # log probability for edge case of 0 (before scaling) 
        log_one_minus_cdf_min = -F.softplus(min_in) # log probability for edge case of 255 (before scaling) 
        #Log probability in the center of the bin
        mid_in = inv_stdv * centered_actions
        log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)
        #Probability for all other cases 
        cdf_delta = cdf_plus - cdf_min 


        #Log probability
        log_probs = torch.where(actions < -0.999, log_cdf_plus,
                        torch.where(actions > 0.999, log_one_minus_cdf_min,
                            torch.where(cdf_delta > 1e-5,
                                torch.log(torch.clamp(cdf_delta, min=1e-12)),
                                    log_pdf_mid - np.log((num_classes - 1) / 2))))
        log_probs = log_probs + F.log_softmax(logit_probs, dim=-1)
        loss = torch.sum(log_sum_exp(log_probs), dim=-1).mean()
        return loss
    
    #Sampling from logistic distribution
    def sample(self, probs, log_scales, means):
        #Selecting Logistic distribution (Sampling from Softmax)
        dist = gumbel_softmax(probs, self.temp) #One hot encoded selection

        #Inversion sampling for logistic mixture sampling
        scales = torch.exp(log_scales) #Make positive
        r1, r2 = 1e-5, 1. - 1e-5
        u = (r1 - r2) * torch.rand(means.shape).cuda() + r2
        actions = means + scales * (torch.log(u) - torch.log(1. - u))
        actions = dist * actions
        actions = actions.sum(dim=-1)

        #Clipping actions within range
        actions = actions.clamp(-0.999, 0.999)
        return actions

    def forward(self, x):
        batch_size = x.shape[0]
        x, _ = self.rnn(x)
        x = x[:, -1] #Get latest output
        probs = self.prob_fc(x)
        means = self.mean_fc(x)
        log_scales = self.log_scale_fc(x)
        # Appropiate dimensions
        probs = probs.view(batch_size, self.out_features, self.n_dist)
        means = means.view(batch_size, self.out_features, self.n_dist)
        log_scales = log_scales.view(batch_size, self.out_features, self.n_dist)
        return probs, log_scales, means