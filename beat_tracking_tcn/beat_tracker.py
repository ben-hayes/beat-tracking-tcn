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
TRIM_SIZE = (81, 3000)
SR = 22050

DEFAULT_CHECKPOINT_PATH = os.path.join(
        os.path.dirname(__file__),
        'checkpoints/default_checkpoint.torch')
DEFAULT_DOWNBEAT_CHECKPOINT_PATH = os.path.join(
        os.path.dirname(__file__),
        'checkpoints/default_downbeat_checkpoint.torch')


model = BeatNet()
model.eval()
downbeat_model = BeatNet(downbeats=True)
downbeat_model.eval()

dbn = DBNBeatTrackingProcessor(
    min_bpm=55,
    max_bpm=215,
    transition_lambda=100,
    fps=(SR / HOP_LENGTH_IN_SAMPLES),
    online=True)
downbeat_dbn = DBNBeatTrackingProcessor(
    min_bpm=10,
    max_bpm=75,
    transition_lambda=100,
    fps=(SR / HOP_LENGTH_IN_SAMPLES),
    online=True)


def beat_activations_from_spectrogram(spectrogram, checkpoint_file=None, downbeats=True):
    if checkpoint_file is not None:
        load_checkpoint(
            downbeat_model if downbeats else model,
            checkpoint_file)
    else:
        load_checkpoint(
            downbeat_model if downbeats else model,
            DEFAULT_DOWNBEAT_CHECKPOINT_PATH
                if downbeats else DEFAULT_CHECKPOINT_PATH)

    with torch.no_grad():
        if type(spectrogram) is not torch.Tensor:
            spectrogram_tensor = torch.from_numpy(spectrogram)\
                                    .unsqueeze(0)\
                                    .unsqueeze(0)\
                                    .float()
        else:
            spectrogram_tensor = spectrogram
        
        return downbeat_model(spectrogram_tensor).numpy() if downbeats\
               else model(spectrogram_tensor).numpy()

def predict_beats_from_spectrogram(
        spectrogram,
        checkpoint_file=None,
        downbeats=True):

    raw_activations =\
        beat_activations_from_spectrogram(
            spectrogram,
            checkpoint_file,
            downbeats).squeeze()
    if downbeats:
        beat_activations = raw_activations[0]
        downbeat_activations = raw_activations[1]

        dbn.reset()
        predicted_beats = dbn.process_offline(beat_activations.squeeze())
        downbeat_dbn.reset()
        predicted_downbeats =\
            downbeat_dbn.process_offline(downbeat_activations.squeeze())

        return predicted_beats, predicted_downbeats
    else:
        beat_activations = raw_activations
        dbn.reset()
        predicted_beats = dbn.process_offline(beat_activations.squeeze())
        return predicted_beats



def beatTracker(input_file, checkpoint_file=None, downbeats=True):
    mag_spectrogram = trim_spectrogram(
        create_spectrogram(
            input_file,
            FFT_SIZE,
            HOP_LENGTH_IN_SECONDS,
            N_MELS),
        TRIM_SIZE).T
    
    return predict_beats_from_spectrogram(
        mag_spectrogram,
        checkpoint_file,
        downbeats)