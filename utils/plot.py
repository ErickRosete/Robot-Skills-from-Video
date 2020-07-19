import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def plot_logistic_distribution(y, act, num_classes=256, n_mix=10, path='./runs'):
    summary_name = path+"/logistic_action_dists"+ datetime.today().strftime('_%m_%d__%H_%M')
    writer = SummaryWriter(summary_name)
    
    y_first = y[0].data.cpu().numpy() # only plotting for first entry
    y_first = np.transpose(y_first, (1, 0))
    act_first = act[0].data.cpu().numpy()

    logit_probs = y_first[:, :n_mix]
    means = y_first[:, n_mix:2 * n_mix]
    scales = scales = y_first[:, 2 * n_mix:3 * n_mix]

    x = np.linspace(-1, 1, num_classes)

    img_batch = []
    for act_idx in range(y_first.shape[0]):
        fig, ax = plt.subplots(1, 1)
        labels = []
        markers_on = [12, 17, 18, 19]
        for i in range (n_mix):
             # plot the predicted mean, scale
            ax.plot(x, stats.logistic.pdf(x, means[act_idx][i], scales[act_idx][i]), label="line 1", markevery=markers_on)
            ax.set_title('action_{0}'.format(act_idx))
            
            # plot the predicted action
            ax.plot(act_first[act_idx], 0, 'bo')
            
            #ax.set_xlabel('action scale')
            #ax.set_ylabel('pdf')
            #labels.append('m='+str(round(means[act_idx][i].item(), 2)) + ', s='+str(round(scales[act_idx][i].item(), 2)))
        #ax.legend(labels, loc="upper right")
        fig.canvas.draw()
        img_batch.append(np.array(fig.canvas.renderer.buffer_rgba())[:,:,:3])
        #fig.savefig('logistic_dist_act_{0}.png'.format(act_idx))
        plt.close()

    img_batch = np.array(img_batch).transpose((0,3,1,2))
    writer.add_images('logistic_action_dists', np.array(img_batch), 0)
    
    writer.close()

if __name__ == '__main__':
    exp_name = 'test'
    BATCH_SIZE = 32
    N_DOF_ROBOT = 9
    N_MIX = 10
    y = torch.rand(BATCH_SIZE, 3 * N_MIX, N_DOF_ROBOT)
    act = torch.rand(BATCH_SIZE, N_DOF_ROBOT)
    
    plot_logistic_distribution(y, act, path='../runs')
