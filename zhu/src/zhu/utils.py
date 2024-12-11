import numpy as np
import torch
import torch.nn as nn
import os
from scipy.signal import butter, lfilter, iirnotch, filtfilt
from zhu.architecture import EEGTransformerNet

def load_model(window_size_sec, fs):
    model = EEGTransformerNet(nb_classes=2, sequence_length=int(window_size_sec*fs), eeg_chans=19)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model.load_state_dict(torch.load(os.path.join(dir_path, 'model.pth'), weights_only=True))
    return model

def load_thresh():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return float(np.load(os.path.join(dir_path, 'best_thresh.npy')))

class SeizureDataset(nn.Module):
    def __init__(self, data, window_size_sec, fs):
        super(SeizureDataset, self).__init__()
        self.data = data
        self.window_size = int(window_size_sec*fs)
        self.fs = fs
        self.recording_duration = int(data.shape[1] / fs)

        # params for preprocessing
        self.lowcut = 0.5
        self.highcut = 120
        notch_1_b, notch_1_a = iirnotch(1, Q=30, fs=fs)
        notch_60_b, notch_60_a = iirnotch(60, Q=30, fs=fs)
        self.notch_1_b = notch_1_b
        self.notch_1_a = notch_1_a
        self.notch_60_b = notch_60_b
        self.notch_60_a = notch_60_a

        window_idx = np.arange(0, self.recording_duration*self.fs, self.fs).astype(int)
        self.window_idx = window_idx[window_idx < self.recording_duration*self.fs - self.window_size]

    def __len__(self):
        return len(self.window_idx)

    def butter_bandpass_filter(self, data, order=3):
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        # y = filtfilt(b, a, data)
        return y

    def preprocess_clip(self, eeg_clip):
        bandpass_filtered_signal = self.butter_bandpass_filter(eeg_clip, order=3)
        filtered_1_signal = lfilter(self.notch_1_b, self.notch_1_a, bandpass_filtered_signal)
        filtered_60_signal = lfilter(self.notch_60_b, self.notch_60_a, filtered_1_signal)  
        eeg_clip = filtered_60_signal
        return eeg_clip

    def __getitem__(self, idx):
        eeg_clip = self.data[:, self.window_idx[idx]:self.window_idx[idx]+self.window_size]
        x = self.preprocess_clip(eeg_clip)
        return torch.tensor(x)

def get_dataloader(data, window_size_sec=25, fs=256):
    dataset = SeizureDataset(data, window_size_sec, fs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False)
    return dataloader

def predict(model, dataloader, device, thresh=0.5):
    model.eval()  # turn on evaluation mode
    preds = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data = data.float().to(device)
            outputs = model(data)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            predicted = probs[:, 1] > thresh
            preds += predicted.cpu().detach().numpy().tolist()
    return np.array(preds)

def get_predict_mask(y_pred, recording_duration, fs, window_size_sec=25, overlap_sec=0):
    y_predict = np.zeros(int(recording_duration*fs))
    window_size = int(window_size_sec*fs)
    overlap = int(overlap_sec*fs)
    for i in range(window_size_sec, len(y_pred)):
        # for each time point, assign majority of all overlaps
        overlap_left = max(0, i*(window_size-overlap))
        overlap_right = min(len(y_predict), (i+1)*(window_size-overlap))
        majority_vote = np.mean(y_pred[max(0, i-window_size_sec):
                                        min(i, recording_duration)]) > 0.5
        y_predict[int(overlap_left):int(overlap_right)] = majority_vote.astype(int)
    return y_predict