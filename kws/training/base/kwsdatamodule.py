import librosa
import numpy as np
import sys

import torch
import torchaudio
from torchvision.transforms import ToTensor
from pytorch_lightning import  LightningDataModule

from base.speechcommands import SubsetSC

class KWSDataModule(LightningDataModule):
    def __init__(self,batch_size=128, num_workers=0, n_fft=1024, 
                 n_mels=128, win_length=None, hop_length=512, class_dict={}, 
                 **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.class_dict = class_dict

    def prepare_data(self):
                
        try:
            self.train_dataset = SubsetSC('training')
        except Exception as e:
            print(f"Error creating Train Dataset object: {e}")
            sys.exit(1)


        try:                                                        
            self.val_dataset = SubsetSC('validation')
        except Exception as e:
            print(f"Error creating Validation Dataset object: {e}")
            sys.exit(1)

        try:
            self.test_dataset = SubsetSC('testing')   
        except Exception as e:
            print(f"Error creating Testing Dataset object: {e}")
            sys.exit(1)

        _, sample_rate, _, _, _ = self.train_dataset[0]
        self.sample_rate = sample_rate
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                              n_fft=self.n_fft,
                                                              win_length=self.win_length,
                                                              hop_length=self.hop_length,
                                                              n_mels=self.n_mels,
                                                              power=2.0)
        self.resize = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)

    def setup(self, stage=None):
        self.prepare_data()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def pad_sequence(self, batch):
    
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

    def collate_fn(self, batch):
       raise NotImplementedError
