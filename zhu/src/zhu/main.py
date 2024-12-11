import numpy as np
import torch
from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg
from zhu.utils import load_model, load_thresh, get_dataloader, predict, get_predict_mask

def main(edf_file, outFile):
    eeg = Eeg.loadEdfAutoDetectMontage(edfFile=edf_file)

    assert eeg.montage is Eeg.Montage.UNIPOLAR, "Error: Only unipolar montages are supported."

    device = "cuda" if torch.cuda.is_available() else "cpu"

    window_size_sec = 25
    fs = eeg.fs
    overlap_ratio = 1-1/window_size_sec
    overlap_sec = window_size_sec * overlap_ratio

    best_thresh = load_thresh()

    model = load_model(window_size_sec, fs)
    model.to(device)

    recording_duration = int(eeg.data.shape[1] / eeg.fs)

    dataloader = get_dataloader(eeg.data, window_size_sec, fs)

    preds = predict(model, dataloader, device, thresh=best_thresh)
    y_predict = get_predict_mask(preds, recording_duration, fs, window_size_sec, overlap_sec)

    hyp = Annotations.loadMask(y_predict, eeg.fs)
    hyp.saveTsv(outFile)

