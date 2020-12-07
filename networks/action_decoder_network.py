import sys
import sys
import math
import torch
import torch.nn as nn
from pathlib import Path
from torch.distributions import Categorical
sys.path.insert(0, str(Path(__file__).parents[1]))
import utils.constants as constants

class ActionDecoderNetwork(nn.Module):
    def __init__(self, k=constants.N_MIXTURES):
        super(ActionDecoderNetwork,self).__init__()
        input_size = constants.VISUAL_FEATURES + constants.N_DOF_ROBOT + \
                     constants.VISUAL_FEATURES + constants.PLAN_FEATURES
        hidden_size, num_layers = 2048, 2
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='relu', batch_first=True)
        input_size, out_features= 2048, constants.N_DOF_ROBOT
        self.mix_model = MDN(input_size, out_features = out_features, num_gaussians = k)

    def loss(self, *args, **kwargs):
        return self.mix_model.loss(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.mix_model.sample(*args, **kwargs)

    def forward(self, x):
        x, _ = self.rnn(x) # b, s, 2048
        return self.mix_model(x) #pi, sigma, mu

""" Mixture Density Network - Gaussian Mixture Model"""
class MDN(nn.Module):

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(    # priors - Softmax guarantees sum = 1
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=-1)
        )
        self.sigma = nn.Linear(in_features, out_features*num_gaussians) # Assume covariance matrix without cross-correlations
        self.mu = nn.Linear(in_features, out_features*num_gaussians)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        pi = self.pi(x)  #b, s, num_gaussians
        sigma = torch.exp(self.sigma(x)) # Guarantees that sigma is positive
        sigma = sigma.view(batch_size, seq_len, self.num_gaussians, self.out_features) # b, s, k, o
        mu = self.mu(x)
        mu = mu.view(batch_size, seq_len, self.num_gaussians, self.out_features) # b, s, k, o
        return pi, sigma, mu

    def loss(self, pi, sigma, mu, target):
        prob = pi * self.gaussian_probability(sigma, mu, target) #b, s, k
        nll = -torch.log(torch.sum(prob, dim=-1)) #b, s
        return torch.mean(nll)

    def gaussian_probability(self, sigma, mu, target):
        norm_term = 1.0 / math.sqrt(2*math.pi)
        data = target.unsqueeze(2).expand_as(sigma) #b, s, k, o
        ret = norm_term * torch.exp(-0.5 * ((data - mu) / sigma)**2) / sigma
        return torch.prod(ret, -1) # b, s, k

    def sample(self, pi, sigma, mu, eps=3e-3):
        if (pi<=eps).any():                             # Verify appropiate priors
            pi[pi<=eps] += eps                          # Prior probability must be bigger than eps
            pi /= pi.sum(axis=-1, keepdim=True)         # Sum must be 1

        categorical = Categorical(pi) 
        sel_gaussians = categorical.sample().data #b, s
        sel_gaussians = sel_gaussians.view(mu.shape[0], mu.shape[1], 1, 1).expand(
                        -1, -1, -1, self.out_features) #b, s, 1, o
        sel_mu = torch.gather(mu, 2, sel_gaussians).squeeze() #b, s, o
        sel_sigma = torch.gather(sigma, 2, sel_gaussians).squeeze() #b, s, o
        sample = torch.normal(sel_mu, sel_sigma) #b, s, o

        # With reparemeterization trick
        # sample = torch.normal(0, 1, size=sel_mu.shape)
        # sample = sel_mu + sample * sel_sigma
        return sample



if __name__ == "__main__":
    b = constants.TRAIN_BATCH_SIZE
    s = constants.WINDOW_SIZE
    d = constants.VISUAL_FEATURES + constants.N_DOF_ROBOT + \
        constants.VISUAL_FEATURES + constants.PLAN_FEATURES
    o = constants.N_DOF_ROBOT

    test_input = torch.randn(b,s,d)
    action_network = ActionDecoderNetwork(k=5)
    pi, sigma, mu = action_network(test_input)
    actions = action_network.sample(pi, sigma, mu)
    target_actions = torch.randn(b,s,o)
    loss = action_network.loss(pi, sigma, mu, target_actions)
    print(loss)
