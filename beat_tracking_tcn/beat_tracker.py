"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: beat_tracking_tcn/beat_tracker.py

Descrption: The main entry point function for the beat tracker. This can be
imported as follows:

>>> from beat_tracking_tcn.beat_tracker import beatTracker

Then it can be invoked like so:

>>> beats, downbeats = beatTracker(path_to_audio_file)
"""
import os
import pickle

from madmom.features import DBNBeatTrackingProcessor
import torch

from beat_tracking_tcn.models.beat_net import BeatNet
from beat_tracking_tcn.utils.spectrograms import create_spectrogram,\
                                                 trim_spectrogram


def load_checkpoint(model, checkpoint_file):
    """
    Restores a model to a given checkpoint, but loads directly to CPU, allowing
    model to be run on non-CUDA devices.
    """    
    model.load_state_dict(
        torch.load(checkpoint_file, map_location=torch.device('cpu')))


# Some important constants that don't need to be command line params
FFT_SIZE = 2048
HOP_LENGTH_IN_SECONDS = 0.01
HOP_LENGTH_IN_SAMPLES = 220
N_MELS = 81
TRIM_SIZE = (81, 3000)
SR = 22050

# Paths to checkpoints distributed with the beat tracker. It's possible to
# call the below functions with custom checkpoints also.
DEFAULT_CHECKPOINT_PATH = os.path.join(
        os.path.dirname(__file__),
        'checkpoints/default_checkpoint.torch')
DEFAULT_DOWNBEAT_CHECKPOINT_PATH = os.path.join(
        os.path.dirname(__file__),
        'checkpoints/default_downbeat_checkpoint.torch')


# Prepare the models
model = BeatNet()
model.eval()
downbeat_model = BeatNet(downbeats=True)
downbeat_model.eval()

# Prepare the post-processing dynamic Bayesian networks, courtesy of madmom.
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


def beat_activations_from_spectrogram(
    spectrogram,
    checkpoint_file=None,
    downbeats=True):
    """
    Given a spectrogram, use the TCN model to compute a beat activation
    function.
    """

    # Load the appropriate checkpoint
    if checkpoint_file is not None:
        load_checkpoint(
            downbeat_model if downbeats else model,
            checkpoint_file)
    else:
        load_checkpoint(
            downbeat_model if downbeats else model,
            DEFAULT_DOWNBEAT_CHECKPOINT_PATH
                if downbeats else DEFAULT_CHECKPOINT_PATH)

    # Speed up computation by skipping torch's autograd
    with torch.no_grad():
        # Convert to torch tensor if necessary
        if type(spectrogram) is not torch.Tensor:
            spectrogram_tensor = torch.from_numpy(spectrogram)\
                                    .unsqueeze(0)\
                                    .unsqueeze(0)\
                                    .float()
        else:
            # Otherwise use the spectrogram as-is
            spectrogram_tensor = spectrogram

        # Forward the spectrogram through the model. Note there are no size
        # restrictions here, as the model is fully convolutional. 
        return downbeat_model(spectrogram_tensor).numpy() if downbeats\
               else model(spectrogram_tensor).numpy()

def predict_beats_from_spectrogram(
        spectrogram,
        checkpoint_file=None,
        downbeats=True):
    """
    Given a spectrogram, predict a list of beat times using the TCN model and
    a DBN post-processor.
    """
    raw_activations =\
        beat_activations_from_spectrogram(
            spectrogram,
            checkpoint_file,
            downbeats).squeeze()

    # Perform independent post-processing for downbeats
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
    """
    Our main entry point — load an audio file, create a spectrogram and predict
    a list of beat times from it.
    """    
    mag_spectrogram = create_spectrogram(
            input_file,
            FFT_SIZE,
            HOP_LENGTH_IN_SECONDS,
            N_MELS).T
    
    return predict_beats_from_spectrogram(
        mag_spectrogram,
        checkpoint_file,
        downbeats)