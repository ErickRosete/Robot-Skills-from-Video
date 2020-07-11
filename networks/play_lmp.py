from network.plan_proposal_network import PlanProposalNetwork
from network.plan_recognition_network import PlanRecognitionNetwork
from network.action_decoder_network import ActionDecoderNetwork
from network.vision_network import  VisionNetwork
import torch
import os
import numpy as np
import torch.optim as optim

class PlayLMP():
    def __init__(self, lr=2e-4, beta=0.01):
        super(PlayLMP, self).__init__()
        self.plan_proposal = PlanProposalNetwork().cuda()
        self.plan_recognition = PlanRecognitionNetwork().cuda()
        self.action_decoder = ActionDecoderNetwork().cuda()
        self.vision = VisionNetwork().cuda()
        params = list(self.plan_proposal.parameters()) + list(self.plan_recognition.parameters()) \
                 + list(self.action_decoder.parameters()) + list(self.vision.parameters())
        self.optimizer = optim.Adam(params, lr=lr)
        self.beta = beta


    def train_mode(self):
        self.vision.train()
        self.plan_proposal.train()
        self.plan_recognition.train()
        self.action_decoder.train()
    
    def eval_mode(self):
        self.vision.eval()
        self.plan_proposal.eval()
        self.plan_recognition.eval()
        self.action_decoder.eval()

    def to_tensor(self, array):
        return torch.tensor(array, dtype=torch.float, device="cuda")

    def step(self, obs, imgs, acts):
        self.train_mode()
        b, s, c, h, w = imgs.shape
        imgs = self.to_tensor(imgs).reshape(-1, c, h, w)
        encoded_imgs = self.vision(imgs)
        encoded_imgs = encoded_imgs.reshape(b, s, -1)
        obs = self.to_tensor(obs)
        pp_input = torch.cat([encoded_imgs[:, 0], obs[:,0], encoded_imgs[:,-1]], dim=-1)
        pr_input = torch.cat([encoded_imgs, obs], dim=-1)

        pp_dist = self.plan_proposal(pp_input)
        pr_dist = self.plan_recognition(pr_input)
        kl_loss = torch.distributions.kl.kl_divergence(pp_dist, pr_dist).mean()

        plan = pr_dist.rsample()
        seq_size = pr_input.shape[1]
        #plan = plan.unsqueeze(1).expand(-1, seq_size, -1)
        #goal_img = encoded_imgs[:, -1].unsqueeze(1).expand(-1, seq_size, -1)
        #action_input = torch.cat([pr_input, goal_img, plan], dim=-1)
        action_input = torch.cat([pp_input, plan], dim=-1).unsqueeze(1)
        alphas, variances, means= self.action_decoder(action_input)
        acts = self.to_tensor(acts[:, 0])
        al_loss = self.action_decoder.loss(alphas, variances, means, acts)
        total_loss = al_loss + self.beta * kl_loss

        # Gradient step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss

    def predict(self, obs, imgs):
        self.eval_mode()
        with torch.no_grad():
            b, s, c, h, w = imgs.shape
            imgs = self.to_tensor(imgs).reshape(-1, c, h, w)
            encoded_imgs = self.vision(imgs)
            encoded_imgs = encoded_imgs.reshape(b, s, -1)
            obs = self.to_tensor(obs)
            pp_input = torch.cat([encoded_imgs[:, 0], obs, encoded_imgs[:,-1]], dim=-1)
            pp_dist = self.plan_proposal(pp_input)
            plan = pp_dist.sample()
            action_input = torch.cat([pp_input, plan], dim=-1).unsqueeze(1)
            alphas, variances, means = self.action_decoder(action_input)
            action = self.action_decoder.sample(alphas, variances, means)
        return action

    def predict_eval(self, obs, imgs, act):
        p_actions = self.predict(obs, imgs).cpu().detach().numpy().squeeze()
        accuracy = np.mean(np.isclose(p_actions, act, atol=0.2))
        return accuracy

    def save(self, file_name):
        torch.save({'plan_proposal': self.plan_proposal.state_dict(),
                    'plan_recognition' : self.plan_recognition.state_dict(),
                    'action_decoder' : self.action_decoder.state_dict(),
                    'vision' : self.vision.state_dict(),
                }, file_name)

    def load(self, file_name):
        if os.path.isfile(file_name):
            print("=> loading checkpoint... ")
            checkpoint = torch.load(file_name)
            self.plan_proposal.load_state_dict(checkpoint['plan_proposal'])
            self.plan_recognition.load_state_dict(checkpoint['plan_recognition'])
            self.action_decoder.load_state_dict(checkpoint['action_decoder'])
            self.vision.load_state_dict(checkpoint['vision'])
            print("done !")
        else:
            print("no checkpoint found...")