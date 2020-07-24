import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class GaussianPolicyNetwork(nn.Module):
    def __init__(self):
        super(GaussianPolicyNetwork, self).__init__()
        input_size, hidden_size, out_features = 393, 2048, 9
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, out_features)
        self.fc_variance = nn.Linear(hidden_size, out_features)

    def loss(self, mean, variance, action):
        normal_dist = Normal(mean, variance)
        nll = -torch.sum(normal_dist.log_prob(action), dim=1)
        return torch.mean(nll)

    def sample(self, mean, variance):
        normal_dist = Normal(mean, variance)
        sample = normal_dist.sample()
        return sample

    def forward(self, x):
        x = x.squeeze()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        variance = F.softplus(self.fc_variance(x))
        return mean, variance