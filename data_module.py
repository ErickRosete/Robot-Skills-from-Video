import pytorch_lightning as pl
import torch
from torch.utils.data import RandomSampler, DataLoader
from custom_sampler import CustomSampler
from relay_kitchen_dataset import VaryingRelayKitchen, PaddedRelayKitchen, ConstantRelayKitchen

class RelayKitchenDataModule(pl.LightningDataModule):
    def __init__(self, root_data_dir="data", batch_size=16, ws_range=[16,32],  \
                window_size=32, type="fixed"):
        super().__init__()
        self.data_dir = root_data_dir
        self.batch_size = batch_size
        self.window_size = window_size
        self.ws_range = ws_range

        assert type == "fixed" or type == "padded" or type =="varying"
        self.type = type

    def setup(self, stage=None):
        training_dir = self.data_dir + '/training'
        val_dir = self.data_dir + '/validation'
        if self.type == "fixed":
            self.train_dataset = ConstantRelayKitchen(training_dir, self.window_size)
            self.val_dataset = ConstantRelayKitchen(val_dir, self.window_size)
        elif self.type == "padded":
            self.train_dataset = PaddedRelayKitchen(training_dir, ws_range = self.ws_range)
            self.val_dataset = PaddedRelayKitchen(val_dir, ws_range = self.ws_range)
        elif self.type == "varying":
            self.train_dataset = VaryingRelayKitchen(training_dir, ws_range = self.ws_range)
            self.train_sampler = CustomSampler( sampler = RandomSampler(self.train_dataset),
                                                batch_size = self.batch_size,
                                                drop_last = False, ws_range = self.ws_range )
            self.val_dataset = VaryingRelayKitchen(val_dir, ws_range = self.ws_range)
            self.val_sampler = CustomSampler( sampler = RandomSampler(self.val_dataset),
                                              batch_size = self.batch_size,
                                              drop_last = False, ws_range = self.ws_range )
        else:
            raise Exception("Invalid type for Dataloader")

    def train_dataloader(self):
        if self.type == "varying":
            return DataLoader(self.train_dataset, batch_sampler=self.train_sampler, num_workers=4,
                            pin_memory=True)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)

    def val_dataloader(self):
        if self.type == "varying":
            return DataLoader(self.val_dataset, batch_sampler=self.val_sampler, num_workers=4,
                            pin_memory=True)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)

if __name__ == "__main__":
    # Test all dataloader types
    types = ["fixed", "padded", "varying"]
    for type in types:
        print("Testing type: %s" % type)
        module = RelayKitchenDataModule(type=type)
        module.setup()
        val_loader = module.val_dataloader()
        train_loader = module.train_dataloader()

        print("Validation")
        for seq_obs, seq_imgs, seq_acts in val_loader:
            print(seq_obs.shape)

        print("Training")
        for seq_obs, seq_imgs, seq_acts in train_loader:
            print(seq_obs.shape)