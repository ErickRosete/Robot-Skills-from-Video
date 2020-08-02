import numpy as np
import matplotlib.pyplot as plt    
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from preprocessing import read_data, preprocess_data, get_filenames, load_data
from networks.play_lmp import PlayLMP
import utils.constants as constants
import os 
import cv2
np.random.seed(0)

#load around 70 samples from each diff file
def load_all_paths():
    train_filenames = ["friday_topknob_bottomknob_switch_slide_0_path",
                "friday_microwave_topknob_bottomknob_hinge_0_path",
                "friday_microwave_kettle_topknob_switch_0_path",
                "friday_microwave_kettle_topknob_hinge_0_path",
                "friday_microwave_kettle_switch_slide_0_path",
                "friday_microwave_kettle_hinge_slide_0_path",
                "friday_microwave_kettle_bottomknob_slide_0_path",
                "friday_microwave_kettle_bottomknob_hinge_0_path",
                "friday_microwave_bottomknob_switch_slide_0_path",
                "friday_microwave_bottomknob_hinge_slide_0_path",
                "friday_kettle_topknob_switch_slide_0_path",
                "friday_kettle_topknob_bottomknob_slide_1_path",
                "friday_kettle_switch_hinge_slide_0_path",
                "friday_kettle_bottomknob_switch_slide_0_path",
                "friday_kettle_bottomknob_hinge_slide_0_path"
                ]
    #join validation and training data
    data_filenames = ["./data/training/%s.pkl"%data_file for data_file in train_filenames]
    data_filenames.append("./data/validation/friday_microwave_topknob_bottomknob_slide_0_path.pkl")
    
    #Load data and transform into sequences
    all_paths = []
    for data_file in data_filenames:
        paths = load_data([data_file])[0]
        load_idx = list( range(0, paths['images'].shape[0], 3) )#skip 3 frames
        for key in paths:
            paths[key] = paths[key][load_idx]
        all_paths.append(paths)
        del paths
    return all_paths

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

def compute_tsne(batch_size = 32, method = "pp", save_name = "clusters", show=False,\
                 n_filenames=5, plot_n_batches = 1, save_dir = "./analysis/recognition_clusters/",\
                 model_path = "./models/10_logistic_multitask_bestacc.pth",\
                 window_size = constants.WINDOW_SIZE, n_mix =constants.N_MIXTURES, logistic=constants.USE_LOGISTICS):
    #Load data and transform into sequences
    #file_names = get_filenames("./data/validation")[:n_filenames] #shuffled filenames
    #validation_paths = load_data(file_names)
    paths = load_all_paths()

    #reset seeds to always get same clusters
    if(method == "pp"): #plan proposal
        data_obs, data_imgs, _ = preprocess_data(paths, window_size, batch_size, validation=True, reset_seed=True)
    else: #plan recognition
        data_obs, data_imgs, _ = preprocess_data(validation_paths, window_size, batch_size, validation=False, reset_seed=True)
        data_obs, data_imgs = data_obs[:plot_n_batches], data_imgs[:plot_n_batches]
    #print("Observation shape", data_obs[0].shape)

    #initialize model
    model = PlayLMP(num_mixtures=n_mix, use_logistics=logistic)
    model.load(model_path)

    X = []
    plot_n_batches = len(data_obs)
    for i in range(plot_n_batches): #get n batches
        #obs = B, S, O | imgs = B, S, H, W, C         
        #Get plan from recognition network
        if( method =="pp"):
            plan_batch = model.get_pp_plan(data_obs[i], data_imgs[i]).detach().cpu().numpy()
        else:
            plan_batch = model.get_pr_plan(data_obs[i], data_imgs[i]).detach().cpu().numpy() #(batch, 256)
        X.append(plan_batch)

    X = np.concatenate(X, axis=0) # plot_n_batches*batch_size, 256
    #dimensionality reduction
    X = PCA(n_components=50, random_state = 0).fit_transform(X) #(batch, 50)
    X = TSNE(n_components=2, random_state = 0).fit_transform(X) #(batch, 2)
    
    #form clusters
    clusters = DBSCAN(eps=2.5, min_samples=8)
    labels = clusters.fit_predict(X)

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
    
    data_imgs = np.concatenate( data_imgs[:plot_n_batches], axis=0)#plot_n_batches * batch_size , seq_len, img_size
    return labels, data_imgs

def plan_proposal_analysis(model_path="./models/10_logistic_multitask_bestacc.pth"):
    cluster_labels, seq_imgs = compute_tsne(batch_size = 50, method ="pp", \
                                            save_dir = "./analysis/tsne_results/", save_name = "proposal_clusters",\
                                            n_filenames=5, plot_n_batches = 3, model_path= model_path,\
                                            window_size=32, n_mix=10, logistic=True)
    save_imgs(cluster_labels, seq_imgs, save_dir ="./analysis/tsne_results/proposal_clusters/")

def plan_recognition_analysis(model_path="./models/model_b62880.pth"):
    cluster_labels, seq_imgs = compute_tsne(batch_size = 10, method = "pr", save_name = "recognition_clusters",\
                                            n_filenames=3, plot_n_batches = 10, model_path=model_path,\
                                            window_size = 8, n_mix = 1, logistic = False)#pp or pr
    #skip_frames =  Win_size//n_imgs - 1
    try:
        save_imgs(cluster_labels, seq_imgs, skip_frames=9, save_dir ="./analysis/tsne_results/recognition_clusters/")
    except Exception as e:
        print(e)

if __name__ == '__main__':
    plan_proposal_analysis(model_path="./models/10_logistic_multitask_bestacc.pth")
    #plan_recognition_analysis(model_path="./models/model_b62880.pth")