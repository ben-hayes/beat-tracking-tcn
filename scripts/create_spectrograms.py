import librosa
import numpy as np
import os
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(
        description="Process a folder of audio files and output a folder of " +
                    "mel spectrograms as NumPy dumps")

    parser.add_argument(
        "audio_directory",
        type=str
    )
    parser.add_argument(
        "output_directory",
        type=str
    )
    parser.add_argument(
        "-f",
        "--fft_size",
        type=int,
        default=2048,
        help="Size of the FFT (default=2048)"
    )
    parser.add_argument(
        "-l",
        "--hop_length",
        type=float,
        default=0.01,
        help="Hop length in seconds (default=0.01)"
    )
    parser.add_argument(
        "-n",
        "--n_mels",
        type=int,
        default=81,
        help="Number of Mel bins (default=81)"
    )

    return parser.parse_args()


def create_spectrograms(
        audio_dir,
        spectrogram_dir,
        n_fft,
        hop_length_in_seconds,
        n_mels):

    for file in os.scandir(audio_dir):
        if os.path.splitext(file.name)[1] != '.wav':
            continue

        x, sr = librosa.load(file.path)
        hop_length_in_samples = int(np.floor(hop_length_in_seconds * sr))
        spec = librosa.feature.melspectrogram(
            x,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length_in_samples,
            n_mels=n_mels)
        mag_spec = np.abs(spec)
        np.save(os.path.join(spectrogram_dir,
                             os.path.splitext(file.name)[0]), mag_spec)
        print('Saved spectrum for {}'.format(file.name))


if __name__ == '__main__':
    args = parse_args()

    create_spectrograms(
        args.audio_directory,
        args.output_directory,
        args.fft_size,
        args.hop_length,
        args.n_mels)
