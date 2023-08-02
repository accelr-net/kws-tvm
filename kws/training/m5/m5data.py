import librosa
import numpy as np
import sys

import torch
from torchvision.transforms import ToTensor


from base.kwsdatamodule import KWSDataModule

class M5Data(KWSDataModule):

    def __init__(self, batch_size: int, num_workers: int, n_fft: int, n_mels: int, win_length, hop_length:int, class_dict: dict, **kwargs):
        super().__init__(batch_size, num_workers, n_fft, n_mels, win_length, hop_length, class_dict, **kwargs)


    def collate_fn(self, batch):
       
        labels = []
        wavs = []
        tensors = []
        for sample in batch:
            waveform, sample_rate, label, speaker_id, utterance_number = sample
            # ensure that all waveforms are 1sec in length; if not pad with zeros
            if waveform.shape[-1] < sample_rate:
                waveform = torch.cat([waveform, torch.zeros((1, sample_rate - waveform.shape[-1]))], dim=-1)
            elif waveform.shape[-1] > sample_rate:
                waveform = waveform[:,:sample_rate]

            # mel from power to db
            tensors += [self.resize(waveform)]
            labels.append(torch.tensor(self.class_dict[label]))
            wavs.append(waveform)

        tensors = self.pad_sequence(tensors)
        labels = torch.stack(labels)
        wavs = torch.stack(wavs)
   
        return tensors, labels, wavs