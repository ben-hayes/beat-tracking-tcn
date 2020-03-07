import os
import pickle

from madmom.features import DBNBeatTrackingProcessor
import torch
import torch.nn as nn

from beat_tracking_tcn.models.beat_net import BeatNet
from beat_tracking_tcn.utils.spectrograms import create_spectrogram,\
                                                 trim_spectrogram


def load_checkpoint(model, checkpoint_file):
    model.load_state_dict(
        torch.load(checkpoint_file, map_location=torch.device('cpu')))


FFT_SIZE = 2048
HOP_LENGTH_IN_SECONDS = 0.01
HOP_LENGTH_IN_SAMPLES = 220
N_MELS = 81
TRIM_SIZE = (3000, 81)
SR = 22050

DEFAULT_CHECKPOINT_PATH = os.path.join(
        os.path.dirname(__file__),
        'checkpoints/default_checkpoint.torch')


model = BeatNet()
load_checkpoint(model, DEFAULT_CHECKPOINT_PATH)
default_checkpoint_loaded = True
model.eval()

dbn = DBNBeatTrackingProcessor(
    min_bpm=55,
    max_bpm=215,
    transition_lambda=100,
    fps=(SR / HOP_LENGTH_IN_SAMPLES),
    online=True)




def track_beats_from_spectrogram(spectrogram, checkpoint_file=None):
    if checkpoint_file is not None:
        load_checkpoint(model, checkpoint_file)
        default_checkpoint_loaded = False
    elif not default_checkpoint_loaded:
        load_checkpoint(model, DEFAULT_CHECKPOINT_PATH)
        default_checkpoint_loaded = True

    with torch.no_grad():
        if type(spectrogram) is not torch.Tensor:
            spectrogram_tensor = torch.from_numpy(spectrogram)\
                                    .unsqueeze(0)\
                                    .unsqueeze(0)\
                                    .float()
        else:
            spectrogram_tensor = spectrogram
        
        beat_activations = model(spectrogram_tensor).numpy()

        dbn.reset()
        predicted_beats = dbn.process_offline(beat_activations.squeeze())
        return predicted_beats


def beatTracker(input_file, checkpoint_file=None):
    mag_spectrogram = trim_spectrogram(
        create_spectrogram(
            input_file,
            FFT_SIZE,
            HOP_LENGTH_IN_SECONDS,
            N_MELS),
        TRIM_SIZE)
    
    return track_beats_from_spectrogram(mag_spectrogram, checkpoint_file)