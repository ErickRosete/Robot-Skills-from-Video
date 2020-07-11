from network.play_lmp import PlayLMP
from torch.utils.tensorboard import SummaryWriter
from preprocessing import read_data, preprocess_data, get_filenames, load_data

if __name__ == "__main__":
    #Network
    writer = SummaryWriter()
    play_lmp = PlayLMP()
    play_lmp.load("./models/model_b4780.pth")
    
    #Hyperparameters
    epochs = 5
    window_size = 16
    val_batch_size = 128
    batch_size = 8 
    files_to_load = 5 #Files to load during training simultaneously

    #Validation data
    validation_paths = read_data("./data/validation")
    val_obs, val_imgs, val_acts = preprocess_data(validation_paths, window_size, val_batch_size, True)
    val_obs, val_imgs, val_acts = val_obs[:5], val_imgs[:5], val_acts[:5] # Keep only 5 batches (Memory)
    print("Validation, number of batches:", len(val_obs))
    print("Validation, batch size:", val_obs[0].shape[0])

    batch = 0
    best_val_accuracy = 0
    for epoch in range(epochs):
        training_filenames = get_filenames("./data/training")
        while len(training_filenames) > 0:
            curr_filenames = training_filenames[:files_to_load] #Loading 5 training files
            del training_filenames[:files_to_load]
            print("Reading training data ...")
            training_paths = load_data(curr_filenames)
            train_obs, train_imgs, train_acts = preprocess_data(training_paths, window_size, batch_size)
            print("Training, number of batches:", len(train_obs))
            print("Training, batch size:", train_obs[0].shape[0])

            while len(train_obs) > 0:
                batch_obs, batch_imgs, batch_acts = train_obs.pop(), train_imgs.pop(), train_acts.pop()
                #STEP
                training_error = play_lmp.step(batch_obs, batch_imgs, batch_acts)
                #Validation eval
                if(batch % 20 == 0):
                    val_accuracy = 0
                    for i in range(len(val_obs)):
                        val_accuracy += play_lmp.predict_eval(val_obs[i], val_imgs[i], val_acts[i])
                    val_accuracy /= len(val_obs)
                    if(val_accuracy > best_val_accuracy):
                        best_val_accuracy = val_accuracy
                        file_name = "./models/model_b%d.pth"%batch
                        play_lmp.save(file_name)
                    writer.add_scalar('Loss/train', training_error, batch)
                    writer.add_scalar('Accuracy/validation', val_accuracy, batch)
                    print("Batch: %d, training error: %.2f, validation accuracy: %.2f" % (batch, training_error, val_accuracy))
                batch += 1  

            #print("Train cycle ...")
        print("Finished " + str(epoch + 1) + " epoch")