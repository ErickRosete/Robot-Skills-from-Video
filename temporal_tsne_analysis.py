import numpy as np
import matplotlib.pyplot as plt    
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from preprocessing import read_data, path_to_batches, get_filenames, load_data
from networks.play_lmp import PlayLMP
import utils.constants as constants
import os 
import cv2

def load_paths():
    valid_filenames = [ "friday_microwave_topknob_bottomknob_slide_%d_path"%i for i in [3,6,8,9,11] ]
    data_filenames = [ "./data/validation/%s.pkl"%data_file for data_file in valid_filenames ]
    paths = load_data(data_filenames)
    return paths

def save_imgs(labels, images, skip_frames=0, save_dir ="./analysis/recognition_clusters/"):
    #images = [batch, seq_len, 3, 300, 300]
    for label in range(labels.max()+1): # ignore outliers since starts at 0
        cluster_imgs = images[labels == label] # All sequences in the same cluster
        save_inds = np.random.choice(cluster_imgs.shape[0], 25) # Select 25 rand imgs to save
        save_imgs = np.transpose(cluster_imgs[save_inds], (0,1,3,4,2)) # Order channels for CV2 save (25, seq_len, 300, 300, 3)

        #Save each sequence as a single image
        save_imgs = tuple(save_imgs[:, i] for i in range(0, save_imgs.shape[1], skip_frames+1))
        save_imgs = np.concatenate(save_imgs, axis=2) # (25, 300, 600, 3), append to width

        #save_directory
        dirname = save_dir + "cluster_%d/" % label
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        for ind, img in enumerate(save_imgs, 1): 
            save_path = "%simg_%d.png" % (dirname, ind)
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) #save as blue shelfs

def temporal_tsne(model_path):
    save_dir = "./analysis/temporal_tsne/"
    save_name = "clusters"
    n_mix = 10
    use_logistics = True
    show = False

    #Initialize model
    model = PlayLMP(num_mixtures=n_mix, use_logistics=use_logistics)
    model.load(model_path)

    #Load data 
    paths = load_paths()
    #Transform data into plans
    X = []
    path_lengths = []
    all_images = []
    for path in paths:
        path_plans = []
        data_obs, data_imgs, _ = path_to_batches(path, window_size=32, batch_size=64, validation=True)
        for j in range(len(data_obs)):
            plans_batch = model.get_pp_plan(data_obs[j], data_imgs[j]).detach().cpu().numpy()
            path_plans.append(plans_batch)
        path_plans = np.concatenate(path_plans, axis=0)
        path_lengths.append(path_plans.shape[0])
        X.append(path_plans)
        all_images.append(np.concatenate(data_imgs, axis=0))
    all_images = np.concatenate(all_images, axis=0)

    #Find clusters
    X = np.concatenate(X, axis=0)
    X = PCA(n_components=50, random_state=1).fit_transform(X) #(batch, 50)
    X = TSNE(n_components=2, random_state=1).fit_transform(X) #(batch, 2)
    clusters = DBSCAN(eps=2, min_samples=5)
    labels = clusters.fit_predict(X)
    print(labels)
    #Temporal visualization of clusters
    c = 0
    for idx, path_length in enumerate(path_lengths):
        aux_labels = labels[c : c + path_length]
        plt.scatter(np.arange(path_length), np.zeros(path_length) + idx, c=aux_labels, cmap="nipy_spectral")
        c += path_length

    #Plot visuals
    plt.suptitle("Temporal Analysis")
    plt.title("Microwave - Topknob - Bottomknob - Slide")
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    if(show):
        plt.show()

    #Save temporal plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig("%s%s.png"%(save_dir, "Temporal"), dpi=100) #save cluster images

    #Plot clusters
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:,0], X[:,1], c=labels, cmap="nipy_spectral")
    fig.suptitle("Latent space")
    ax.axis('off')
    
    # Plot legends
    fig_legends, ax_legends = plt.subplots(figsize = (3,10))
    n_classes = np.unique(labels)
    legend1 = ax_legends.legend(*scatter.legend_elements(num = n_classes ), title="Classes", loc='center')
    ax_legends.add_artist(legend1)

    #save in analysis/tsne
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    fig.savefig("%s%s.png"%(save_dir,save_name), dpi=100) #save cluster images
    fig_legends.savefig("%s%s_labels.png"%(save_dir,save_name), dpi=100) #save legends

    if(show):
        plt.show()
    
    data_imgs = np.concatenate(data_imgs, axis=0) # batch_size , seq_len, img_size
    return labels, all_images

if __name__ == '__main__':
    cluster_labels, seq_imgs = temporal_tsne(model_path="./models/10_logistic_multitask_bestacc_new.pth")
    save_imgs(cluster_labels, seq_imgs, save_dir ="./analysis/temporal_tsne/proposal_clusters/")
