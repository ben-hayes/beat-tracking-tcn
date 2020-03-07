import os

import librosa
import numpy as np

def create_spectrogram(
        file_path,
        n_fft,
        hop_length_in_seconds,
        n_mels):
    
    x, sr = librosa.load(file_path)
    hop_length_in_samples = int(np.floor(hop_length_in_seconds * sr))
    spec = librosa.feature.melspectrogram(
        x,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length_in_samples,
        n_mels=n_mels)
    mag_spec = np.abs(spec)

    return mag_spec

def create_spectrograms(
        audio_dir,
        spectrogram_dir,
        n_fft,
        hop_length_in_seconds,
        n_mels):

    for file in os.scandir(audio_dir):
        if os.path.splitext(file.name)[1] != '.wav':
            continue

        mag_spec = create_spectrogram(
            file.path,
            n_fft,
            hop_length_in_seconds,
            n_mels)
        np.save(os.path.join(spectrogram_dir,
                             os.path.splitext(file.name)[0]), mag_spec)
        print('Saved spectrum for {}'.format(file.name))

def trim_spectrogram(spectrogram, trim_size):
    output = np.zeros(trim_size)
    dim0_range = min(trim_size[0], spectrogram.shape[0])
    dim1_range = min(trim_size[1], spectrogram.shape[1])

    output[:dim0_range, :dim1_range] = spectrogram[:dim0_range, :dim1_range]
    return output
