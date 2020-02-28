"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: datasets/ballroom_dataset.py
Description: A PyTorch dataset class representing the ballroom dataset.
"""

import torch
from torch.utils.data import Dataset

import os
import numpy as np


class BallroomDataset(Dataset):
    """
    A PyTorch Dataset wrapping the ballroom dataset for beat detection tasks

    Provides mel spectrograms and a vector of beat annotations per spectrogram
    frame as per the Davies & Böck paper (that is, with two frames of 0.5
    either side of the beat's peak.

    Requires dataset to be preprocessed to mel spectrograms using the provided
    script.
    """

    def __init__(
            self,
            spectrogram_dir,
            label_dir,
            sr=22050,
            hop_size_in_seconds=0.01,
            trim_size=(81, 3000)):
        """
        Initialise the dataset object.

        Parameters:
            spectrogram_dir: directory holding spectrograms as NumPy dumps
            label_dir: directory containing labels as NumPy dumps

        Keyword Arguments:
            sr (=22050): Sample rate to use when converting annotations to
                         spectrogram frame indices
            hop_size_in_seconds (=0.01): Mel spectrogram hop size
            trim_size (=(81,3000)): Dimensions to trim spectrogram down to.
                                    Should match input dimensions of network.
        """
        self.spectrogram_dir = spectrogram_dir
        self.label_dir = label_dir
        self.data_names = self._get_data_list()

        self.sr = 22050
        self.hop_size = int(np.floor(hop_size_in_seconds * 22050))
        self.trim_size = trim_size

    def __len__(self):
        """Overload len() calls on object."""
        return len(self.data_names)

    def __getitem__(self, i):
        """Overload square bracket indexing on object"""
        raw_spec, raw_beats = self._load_spectrogram_and_labels(i)
        x, y = self._trim_spec_and_labels(raw_spec, raw_beats)

        return {
            'spectrogram': torch.from_numpy(
                    np.expand_dims(x.T, axis=0)).float(),
            'target': torch.from_numpy(y[:3000].astype('float64')).float(),
        }

    def get_name(self, i):
        """Fetches name of datapoint specified by index i"""
        return self.data_names[i]

    def get_ground_truth(self, i, quantised=True):
        """
        Fetches ground truth annotations for datapoint specified by index i

        Parameters:
            i: Index signifying which datapoint to fetch truth for

        Keyword Arguments:
            quantised (=True): Whether to return a quantised grount truth
        """

        return self._get_quantised_ground_truth(i)\
            if quantised else self._get_unquantised_ground_truth(i)

    def _trim_spec_and_labels(self, spec, labels):
        """
        Trim spectrogram matrix and beat label vector to dimensions specified
        in self.trim_size. Returns tuple of trimmed NumPy arrays

        Parameters:
            spec: Spectrogram as NumPy array
            labels: Labels as NumPy array
        """

        x = np.zeros(self.trim_size)
        y = np.zeros(self.trim_size[1])

        to_x = self.trim_size[0]
        to_y = min(self.trim_size[1], spec.shape[1])

        x[:to_x, :to_y] = spec[:, :to_y]
        y[:to_y] = labels[:to_y]

        return x, y

    def _get_data_list(self):
        """Fetches list of datapoints in label directory"""

        names = []
        for entry in os.scandir(self.label_dir):
            names.append(os.path.splitext(entry.name)[0])
        return names

    def _text_label_to_float(self, text):
        """Exracts beat time from a text line and converts to a float"""
        allowed = '1234567890.'
        t = text.rstrip('\n').split(' ')[0]
        filtered = ''.join([c for c in t if c in allowed])
        return float(filtered)

    def _get_quantised_ground_truth(self, i):
        """
        Fetches the ground truth (time labels) from the appropriate
        label file. Then, quantises it to the nearest spectrogram frames in
        order to allow fair performance evaluation.
        """

        with open(
                os.path.join(self.label_dir, self.data_names[i] + '.beats'),
                'r') as f:

            beat_times =\
                np.array([self._text_label_to_float(line) for line in f])\
                * self.sr

        quantised_times = []

        for time in beat_times:
            spec_frame = int(time / self.hop_size)
            quantised_time = spec_frame * self.hop_size / self.sr
            quantised_times.append(quantised_time)

        return np.array(quantised_times)

    def _get_unquantised_ground_truth(self, i):
        """
        Fetches the ground truth (time labels) from the appropriate
        label file.
        """

        with open(
                os.path.join(self.label_dir, self.data_names[i] + '.beats'),
                'r') as f:

            beat_times =\
                np.array([self._text_label_to_float(line) for line in f])

        return beat_times

    def _load_spectrogram_and_labels(self, i):
        """
        Given an index for the data name array, return the contents of the
        corresponding spectrogram and label dumps.
        """
        data_name = self.data_names[i]

        with open(
                os.path.join(self.label_dir, data_name + '.beats'),
                'r') as f:

            beat_times = np.array(
                [self._text_label_to_float(line) for line in f]) * self.sr

        spectrogram =\
            np.load(os.path.join(self.spectrogram_dir, data_name + '.npy'))
        beat_vector =\
            np.zeros(spectrogram.shape[-1])

        for time in beat_times:
            spec_frame =\
                min(int(time / self.hop_size), beat_vector.shape[0] - 1)
            for n in range(-2, 3):
                if 0 <= spec_frame + n < beat_vector.shape[0]:
                    beat_vector[spec_frame + n] = 1.0 if n == 0 else 0.5

        return spectrogram, beat_vector
