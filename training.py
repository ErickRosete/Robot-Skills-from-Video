from networks.play_lmp import PlayLMP
from torch.utils.tensorboard import SummaryWriter
from preprocessing import read_data, preprocess_data, get_filenames, load_data, mult_wind_preprocessing
import utils.constants as constants
import numpy as np
from datetime import datetime
import argparse

if __name__ == "__main__":
    #----------- Parser ------------#
    parser = argparse.ArgumentParser(description='some description')
    parser.add_argument('--exp_name', dest='exp_name', type=str, default='mws_10_logistic_multitask')
    args = parser.parse_args()
    print(args)
    #-------------------------------#
    exp_name = args.exp_name
    summary_name = "./runs/"+ exp_name + datetime.today().strftime('_%m_%d__%H_%M') # month, day, hr, min
    # ------------ Initialization ------------ #
    writer = SummaryWriter(summary_name)
    play_lmp = PlayLMP(constants.LEARNING_RATE, constants.BETA, \
                       constants.N_MIXTURES, constants.USE_LOGISTICS)
    #play_lmp.load("./models/model_b4780.pth")
    
    # ------------ Hyperparams ------------ #
    epochs = constants.N_EPOCH
    min_ws = constants.MIN_WINDOWS_SIZE
    max_ws = constants.MAX_WINDOW_SIZE
    val_batch_size = constants.VAL_BATCH_SIZE
    batch_size = constants.TRAIN_BATCH_SIZE 
    files_to_load = constants.FILES_TO_LOAD #Files to load during training simultaneously
    eval_freq = constants.EVAL_FREQ #validate every eval_freq batches

    # ------------ Validation data loading ------------ #
    # Validation data
    validation_paths = read_data("./data/validation")
    #val_obs, val_imgs, val_acts = preprocess_data(validation_paths, window_size, val_batch_size, True)
    val_obs, val_imgs, val_acts = mult_wind_preprocessing(validation_paths, min_ws, max_ws, val_batch_size)
    #Note when preprocesing validation data we return 
    #[current_img, goal_img] , current_obs, current_action
    #then val_imgs=(batch,2,3,300,300), val_acts = (batch,9), val_obs = (batch,9)
    val_obs, val_imgs, val_acts = val_obs[:5], val_imgs[:5], val_acts[:5] # Keep only 5 batches (Memory)
    print("Validation, number of batches:", len(val_obs))
    print("Validation, batch size:", val_obs[0].shape[0])

    # ------------ Training ------------ #
    batch = 0
    best_val_accuracy, best_val_loss = 0, float('inf')
    for epoch in range(epochs):
        training_filenames = get_filenames("./data/training")
        # ------------ Filenames loop ------------ #
        while len(training_filenames) > 0:
            # ------------ Load training data ------------ #
            curr_filenames = training_filenames[:files_to_load] #Loading 5 training files
            del training_filenames[:files_to_load]
            print("Reading training data ...")
            training_paths = load_data(curr_filenames)
            window_size = np.random.randint(min_ws, max_ws + 1)
            #window_size = constants.WINDOW_SIZE
            train_obs, train_imgs, train_acts = preprocess_data(training_paths, window_size, batch_size)
            print("Training, number of batches:", len(train_obs))
            print("Training, batch size:", train_obs[0].shape[0])

            # ------------ Batch training loop ------------ #
            while len(train_obs) > 0:
                batch_obs, batch_imgs, batch_acts = train_obs.pop(), train_imgs.pop(), train_acts.pop()
                # STEP
                training_error, mix_loss, kl_loss = play_lmp.step(batch_obs, batch_imgs, batch_acts)
                
                # ------------ Evaluation ------------ #
                if(batch % eval_freq == 0):
                    val_accuracy, val_mix_loss = 0, 0
                    # For every batch in val_data
                    for i in range(len(val_obs)):
                        batch_accuracy, batch_mix_loss = play_lmp.predict_eval(val_obs[i], val_imgs[i], val_acts[i])
                        val_mix_loss += batch_mix_loss
                        val_accuracy += batch_accuracy
                    val_accuracy /= len(val_obs)
                    val_mix_loss /= len(val_obs)
                    #Save only the best models
                    if(val_accuracy > best_val_accuracy):
                        best_val_accuracy = val_accuracy
                        file_name = "./models/%s_b%d_bv.pth" % (exp_name, batch)
                        play_lmp.save(file_name)
                    
                    if(val_mix_loss < best_val_loss):
                        best_val_loss = val_mix_loss
                        file_name = "./models/%s_b%d_bl.pth" % (exp_name, batch)
                        play_lmp.save(file_name)
                    
                    #Log to tensorboard
                    writer.add_scalar('train/total_loss', training_error, batch)
                    writer.add_scalar('train/mixture_loss', mix_loss, batch)
                    writer.add_scalar('train/KL_div', kl_loss, batch)
                    writer.add_scalar('validation/mixture_loss', val_mix_loss, batch)
                    writer.add_scalar('validation/accuracy', val_accuracy, batch)
                    print("Batch: %d, training error: %.2f, validation accuracy: %.2f" % (batch, training_error, val_accuracy))
                batch += 1  

            #print("Train cycle ...")
        print("Finished " + str(epoch + 1) + " epoch")