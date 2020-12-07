from networks.play_lmp import PlayLMP
from torch.utils.tensorboard import SummaryWriter
from data_module import RelayKitchenDataModule
import utils.constants as constants
import numpy as np
from datetime import datetime
import argparse

if __name__ == "__main__":
    #----------- Parser ------------#
    parser = argparse.ArgumentParser(description='some description')
    parser.add_argument('--exp_name', dest='exp_name', type=str, default='10_logistic_multitask')
    args = parser.parse_args()
    print(args)
    #-------------------------------#
    exp_name = args.exp_name
    summary_name = "./runs/"+ exp_name + datetime.today().strftime('_%m_%d__%H_%M') # month, day, hr, min
    # ------------ Initialization ------------ #
    writer = SummaryWriter(summary_name)
    play_lmp = PlayLMP(constants.LEARNING_RATE, constants.BETA, \
                       constants.N_MIXTURES, constants.USE_LOGISTICS)
    
    # ------------ Hyperparams ------------ #
    epochs = constants.N_EPOCH
    window_size = constants.WINDOW_SIZE
    min_ws = constants.MIN_WINDOWS_SIZE
    max_ws = constants.MAX_WINDOW_SIZE
    val_batch_size = constants.VAL_BATCH_SIZE
    batch_size = constants.TRAIN_BATCH_SIZE 
    files_to_load = constants.FILES_TO_LOAD #Files to load during training simultaneously
    eval_freq = constants.EVAL_FREQ #validate every eval_freq batches

    # ------------ Validation data loading ------------ #
    # Validation data
    module = RelayKitchenDataModule(type="fixed", window_size=window_size, batch_size=batch_size)
    module.setup()
    val_loader = module.val_dataloader()
    train_loader = module.train_dataloader()

    # ------------ Training ------------ #
    batch = 0
    best_val_accuracy, best_val_loss = 0, float('inf')
    for epoch in range(epochs):
        for batch_obs, batch_imgs, batch_acts in train_loader:
                training_error, mix_loss, kl_loss = play_lmp.step(batch_obs, batch_imgs, batch_acts)
                                
                # ------------ Evaluation ------------ #
                if(batch % eval_freq == 0):
                    val_accuracy, val_mix_loss = 0, 0
                    # For every batch in val_data
                    for val_obs, val_imgs, val_acts in val_loader:
                        batch_accuracy, batch_mix_loss = play_lmp.predict_eval(val_obs, val_imgs, val_acts)
                        val_mix_loss += batch_mix_loss
                        val_accuracy += batch_accuracy
                    val_accuracy /= len(val_loader)
                    val_mix_loss /= len(val_loader)
                    #Save only the best models
                    if(val_accuracy > best_val_accuracy):
                        best_val_accuracy = val_accuracy
                        file_name = "./models/%s_bestacc.pth" % (exp_name)
                        play_lmp.save(file_name)
                        
                    if(val_mix_loss < best_val_loss):
                        best_val_loss = val_mix_loss
                        file_name = "./models/%s_bestloss.pth" % (exp_name)
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
