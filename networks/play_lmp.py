#import from training dir
from networks.networks import  VisionNetwork, PlanRecognitionNetwork, PlanProposalNetwork, LogisticPolicyNetwork
from networks.action_decoder_network import ActionDecoderNetwork
from networks.gaussian_policy_network import GaussianPolicyNetwork
import torch
import os
import numpy as np
import torch.optim as optim
import torch.distributions as D
from torch.distributions.normal import Normal
import utils.mixture as mixtures

class PlayLMP():
    def __init__(self, lr=2e-4, beta=0.01, num_mixtures=5, use_logistics=False):
        super(PlayLMP, self).__init__()
        self.plan_proposal = PlanProposalNetwork().cuda()
        self.plan_recognition = PlanRecognitionNetwork().cuda()
        self.vision = VisionNetwork().cuda()
        self.num_mixtures = num_mixtures
        self.use_logistics = use_logistics
        if use_logistics:
            self.action_decoder = LogisticPolicyNetwork(num_mixtures).cuda() #default n_mix = 10
        elif(num_mixtures > 1):
            self.action_decoder = ActionDecoderNetwork(num_mixtures).cuda()
        else:
            self.action_decoder = GaussianPolicyNetwork().cuda()
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

    def get_pp_plan(self, obs, imgs):
        #obs = (batch_size, 9)
        #imgs = (batch_size, 2, 3, 300, 300)
        self.eval_mode()
        with torch.no_grad():
            b, s, c, h, w = imgs.shape
            imgs = self.to_tensor(imgs).reshape(-1, c, h, w)
            # ------------ Vision Network ------------ #
            encoded_imgs = self.vision(imgs)
            encoded_imgs = encoded_imgs.reshape(b, s, -1)

            # ------------Plan Proposal------------ #
            obs = self.to_tensor(obs)
            pp_input = torch.cat([encoded_imgs[:, 0], obs, encoded_imgs[:,-1]], dim=-1)
            mu_p, sigma_p = self.plan_proposal(pp_input)#(batch, 256) each
            pp_dist = Normal(mu_p, sigma_p)
            sampled_plan = pp_dist.sample()
        return sampled_plan

    def get_pr_plan(self, obs, imgs):
        #inputs are np arrays
        #obs = (batch_size, seq_len, 9)
        #imgs = (batch_size, seq_len , 3, 300, 300)
        self.eval_mode()
        with torch.no_grad():
            b, s, c, h, w = imgs.shape
            imgs = self.to_tensor(imgs).reshape(-1, c, h, w)
            # ------------ Vision Network ------------ #
            encoded_imgs = self.vision(imgs)
            encoded_imgs = encoded_imgs.reshape(b, s, -1)

            # ------------Plan Recognition------------ #
            #plan recognition input = visuo_proprio =  (batch_size, sequence_length, 73)
            obs = self.to_tensor(obs)
            pr_input = torch.cat([encoded_imgs, obs], dim=-1)
            mu_p, sigma_p = self.plan_recognition(pr_input)#(batch, 256) each
            pr_dist = Normal(mu_p, sigma_p)
            sampled_plan = pr_dist.sample()
        return sampled_plan

    #Forward + loss + backward
    def step(self, obs, imgs, acts):
        self.train_mode()
        b, s, c, h, w = imgs.shape
        imgs = self.to_tensor(imgs).reshape(-1, c, h, w) #(batch_size * sequence_length, 3, 300, 300)

        # ------------ Vision Network ------------ #
        encoded_imgs = self.vision(imgs) #(batch*seq_len, 64)
        encoded_imgs = encoded_imgs.reshape(b, s, -1) #(batch, seq, 64)

        # ------------Plan Proposal------------ #
        #plan proposal input = cat(visuo_proprio, goals) = (batch, 137)
        obs = self.to_tensor(obs)
        pp_input = torch.cat([encoded_imgs[:, 0], obs[:,0], encoded_imgs[:,-1]], dim=-1)
        mu_p, sigma_p = self.plan_proposal(pp_input)#(batch, 256) each
        pp_dist = Normal(mu_p, sigma_p)

        # ------------Plan Recognition------------ #
        #plan proposal input = visuo_proprio =  (batch_size, sequence_length, 73)
        pr_input = torch.cat([encoded_imgs, obs], dim=-1)
        mu_r, sigma_r = self.plan_recognition(pr_input)#(batch, 256) each
        pr_dist = Normal(mu_r, sigma_r)

        # ------------ Policy network ------------ #
        sampled_plan = pr_dist.rsample() #sample from recognition net
        action_input = torch.cat([pp_input, sampled_plan], dim=-1).unsqueeze(1)
        if self.use_logistics:
            logit_probs, scales, means = self.action_decoder(action_input)
        elif(self.num_mixtures > 1):
            alphas, variances, means= self.action_decoder(action_input)
        else:
            mean, variance = self.action_decoder(action_input)
        acts = self.to_tensor(acts[:, 0])

        # ------------ Loss ------------ #
        kl_loss = D.kl_divergence(pr_dist, pp_dist).mean()
        if self.use_logistics:
            prediction = torch.cat([logit_probs, means, scales], dim=1)
            target = acts.unsqueeze(2)
            mix_loss = mixtures.discretized_mix_logistic_loss(target, prediction, reduce=True)
        elif(self.num_mixtures > 1):
            mix_loss = self.action_decoder.loss(alphas, variances, means, acts)
        else:
            mix_loss = self.action_decoder.loss(mean, variance, acts)
        total_loss = mix_loss + self.beta * kl_loss

        # ------------ Backward pass ------------ #
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss, mix_loss, kl_loss

    #Evaluation in test set, no grad, no labels
    def predict(self, obs, imgs):
        self.eval_mode()
        with torch.no_grad():
            b, s, c, h, w = imgs.shape
            imgs = self.to_tensor(imgs).reshape(-1, c, h, w)
            # ------------ Vision Network ------------ #
            encoded_imgs = self.vision(imgs)
            encoded_imgs = encoded_imgs.reshape(b, s, -1)

            # ------------Plan Proposal------------ #
            obs = self.to_tensor(obs)
            pp_input = torch.cat([encoded_imgs[:, 0], obs, encoded_imgs[:,-1]], dim=-1)
            mu_p, sigma_p = self.plan_proposal(pp_input)#(batch, 256) each
            pp_dist = Normal(mu_p, sigma_p)

            # ------------ Policy network ------------ #
            sampled_plan = pp_dist.sample() #sample from proposal net
            action_input = torch.cat([pp_input, sampled_plan], dim=-1).unsqueeze(1)
            if self.use_logistics:
                logit_probs, scales, means = self.action_decoder(action_input)
                prediction = torch.cat([logit_probs, means, scales], dim=1)
                action = mixtures.sample_from_discretized_mix_logistic(prediction)
            elif(self.num_mixtures > 1):
                alphas, variances, means= self.action_decoder(action_input)
                action = self.action_decoder.sample(alphas, variances, means)
            else:
                mean, variance = self.action_decoder(action_input)
                action = self.action_decoder.sample(mean, variance)

        return action

    #Predict method to be able to compute val accuracy and error.
    #inputs: numpy arrays (Batch, seq_len, dim)
    def predict_eval(self, obs, imgs, act):
        self.eval_mode()
        with torch.no_grad():
            b, s, c, h, w = imgs.shape
            imgs = self.to_tensor(imgs).reshape(-1, c, h, w)
            # ------------ Vision Network ------------ #
            encoded_imgs = self.vision(imgs)
            encoded_imgs = encoded_imgs.reshape(b, s, -1)

            # ------------Plan Proposal------------ #
            obs = self.to_tensor(obs)
            pp_input = torch.cat([encoded_imgs[:, 0], obs, encoded_imgs[:,-1]], dim=-1)
            mu_p, sigma_p = self.plan_proposal(pp_input)#(batch, 256) each
            pp_dist = Normal(mu_p, sigma_p)

            # ------------ Policy network ------------ #
            sampled_plan = pp_dist.sample() #sample from proposal net
            action_input = torch.cat([pp_input, sampled_plan], dim=-1).unsqueeze(1)
            if self.use_logistics:
                logit_probs, scales, means = self.action_decoder(action_input)
                prediction = torch.cat([logit_probs, means, scales], dim=1)
                action = mixtures.sample_from_discretized_mix_logistic(prediction)
            elif(self.num_mixtures > 1):
                alphas, variances, means= self.action_decoder(action_input)
                action = self.action_decoder.sample(alphas, variances, means)
            else:
                mean, variance = self.action_decoder(action_input)
                action = self.action_decoder.sample(mean, variance)

            # ------------ Loss ------------ #
            #cannot compute KL_divergence, only return mixture loss
            action_labels = self.to_tensor(act)
            if self.use_logistics:
                prediction = torch.cat([logit_probs, means, scales], dim=1)
                target = action_labels.unsqueeze(2)
                mix_loss = mixtures.discretized_mix_logistic_loss(target, prediction, reduce=True)
            elif(self.num_mixtures > 1):
                mix_loss = self.action_decoder.loss(alphas, variances, means, action_labels)
            else:
                mix_loss = self.action_decoder.loss(mean, variance, action_labels)
            
            # ------------ Accuracy ------------ #
            p_actions = action.cpu().detach().numpy().squeeze()
            accuracy = np.mean(np.isclose(p_actions, act, atol=0.2))
        return accuracy, mix_loss

    def predict_with_plan(self, obs, imgs, plan):
        with torch.no_grad():
            b, s, c, h, w = imgs.shape
            imgs = self.to_tensor(imgs).reshape(-1, c, h, w)
            # ------------ Vision Network ------------ #
            encoded_imgs = self.vision(imgs)
            encoded_imgs = encoded_imgs.reshape(b, s, -1)

            # ------------Plan Proposal------------ #
            obs = self.to_tensor(obs)
            pp_input = torch.cat([encoded_imgs[:, 0], obs, encoded_imgs[:,-1]], dim=-1)
            action_input = torch.cat([pp_input, plan], dim=-1).unsqueeze(1)
            if self.use_logistics:
                logit_probs, scales, means = self.action_decoder(action_input)
                prediction = torch.cat([logit_probs, means, scales], dim=1)
                action = mixtures.sample_from_discretized_mix_logistic(prediction)
            elif(self.num_mixtures > 1):
                alphas, variances, means= self.action_decoder(action_input)
                action = self.action_decoder.sample(alphas, variances, means)
            else:
                mean, variance = self.action_decoder(action_input)
                action = self.action_decoder.sample(mean, variance)

        return action

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
