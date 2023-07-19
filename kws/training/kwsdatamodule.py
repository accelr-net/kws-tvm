import librosa
import numpy as np
import sys

import torch
import torchaudio
from torchvision.transforms import ToTensor
from pytorch_lightning import  LightningDataModule

from speechcommands import SubsetSC

class KWSDataModule(LightningDataModule):
    def __init__(self,batch_size=128, num_workers=0, n_fft=512, 
                 n_mels=128, win_length=None, hop_length=256, class_dict={}, 
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

    def collate_fn(self, batch):
        mels = []
        labels = []
        wavs = []
        for sample in batch:
            waveform, sample_rate, label, speaker_id, utterance_number = sample
            # ensure that all waveforms are 1sec in length; if not pad with zeros
            if waveform.shape[-1] < sample_rate:
                waveform = torch.cat([waveform, torch.zeros((1, sample_rate - waveform.shape[-1]))], dim=-1)
            elif waveform.shape[-1] > sample_rate:
                waveform = waveform[:,:sample_rate]

            # mel from power to db
            mels.append(ToTensor()(librosa.power_to_db(self.transform(waveform).squeeze().numpy(), ref=np.max)))
            labels.append(torch.tensor(self.class_dict[label]))
            wavs.append(waveform)

        mels = torch.stack(mels)
        labels = torch.stack(labels)
        wavs = torch.stack(wavs)
   
        return mels, labels, wavs