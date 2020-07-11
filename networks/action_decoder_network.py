import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import Categorical
from torch.autograd import Variable
import math

ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)
class ActionDecoderNetwork(nn.Module):
    def __init__(self, k=5):
        super(ActionDecoderNetwork,self).__init__()
        self.k = k
        input_size, hidden_size, num_layers = 393, 2048, 2
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='relu', batch_first=True)
        input_size, out_features= 2048, 9
        self.mix_model = MDN(input_size, out_features = out_features, num_gaussians = k)

    def loss(self, pi, sigma, mu, target):
        prob = pi * self.gaussian_probability(sigma, mu, target)
        nll = -torch.log(torch.sum(prob, dim=1))
        return torch.mean(nll)

    def gaussian_probability(self, sigma, mu, target):
        data = target.unsqueeze(1).expand_as(sigma)
        ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((data - mu) / sigma)**2) / sigma
        return torch.prod(ret, 2)

    def sample(self, pi, sigma, mu, eps=1e-5):
        categorical = Categorical(pi + eps)
        pis = list(categorical.sample().data)
        sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
        for i, idx in enumerate(pis):
            sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx])
        return sample

    def forward(self, x):
        batch_size = x.shape[0]
        x, h= self.rnn(x) #(batch,1,2048)  
        x = x.view(batch_size, -1) # (batch,2048)  
        pi, sigma, mu = self.mix_model(x) #pi = (batch, k) #sigma = (batch, k, 9) #mu = (batch, k, 9)
        return pi, sigma, mu



class MDN(nn.Module):

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=-1)
        )
        self.sigma = nn.Linear(in_features, out_features*num_gaussians)
        self.mu = nn.Linear(in_features, out_features*num_gaussians)

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu