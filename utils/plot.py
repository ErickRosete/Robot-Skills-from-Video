import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def plot_logistic_distribution(y, writer, num_classes=256, n_mix=10):
    y = np.transpose(y, (0, 2, 1))
    first = y[0] # only plotting for first entry

    logit_probs = first[:, :n_mix]
    means = first[:, n_mix:2 * n_mix]
    scales = scales = first[:, 2 * n_mix:3 * n_mix]

    x = np.linspace(-1, 1, num_classes)

    img_batch = []
    for act_idx in range(first.shape[0]):
        fig, ax = plt.subplots(1, 1)
        labels = []
        for i in range (n_mix):
            ax.plot(x, stats.logistic.pdf(x, means[act_idx][i], scales[act_idx][i]), label="line 1")
            ax.set_title('action_{0}'.format(act_idx))
            #ax.set_xlabel('action scale')
            #ax.set_ylabel('pdf')
            #labels.append('m='+str(round(means[act_idx][i].item(), 2)) + ', s='+str(round(scales[act_idx][i].item(), 2)))
        #ax.legend(labels, loc="upper right")
        fig.canvas.draw()
        img_batch.append(np.array(fig.canvas.renderer.buffer_rgba())[:,:,:3])
        #fig.savefig('logistic_dist_act_{0}.png'.format(act_idx))

    img_batch = np.array(img_batch).transpose((0,3,1,2))
    writer.add_images('logistic_action_dists', np.array(img_batch), 0)

if __name__ == '__main__':
    exp_name = 'test'
    BATCH_SIZE = 32
    N_DOF_ROBOT = 9
    N_MIX = 10
    y = torch.rand(BATCH_SIZE, 3 * N_MIX, N_DOF_ROBOT)

    summary_name = "./runs/"+ exp_name +'logistic_action_dists'+ datetime.today().strftime('_%m_%d__%H_%M')
    writer = SummaryWriter(summary_name)
    plot_logistic_distribution(y, writer)
    writer.close()
