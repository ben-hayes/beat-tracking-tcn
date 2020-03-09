# Beat Tracking TCN

An implementation of Davies &amp; Böck's beat-tracking temporal convolutional network [1].

## Usage

In order to use the beat tracker, this package and its dependencies must be installed with pip. It is recommended to do so in a virtualenv as follows:
```bash
python -m venv venv
source venv/bin/activate
```

The dependencies can then be installed like so:
```bash
pip install -r requirements.txt
```

Once this is done, the beat tracker can be accessed like this:
```python
from beat_tracking_tcn.beat_tracker import beatTracker

beats, downbeats = beatTracker("path_to_audio_file")
```

Alternatively, if you have the required dependencies `librosa`, `madmom`, `mir-eval`, `numpy`, and `torch` installed system-wide, you can perform a system wide install by running the following command from the root of this source repo:
```bash
pip install -e .
```

The beat tracker can then be invoked as above.

## References

[1] M. E. P. Davies and S. Bock, _‘Temporal convolutional networks for musical audio beat tracking’_, in 2019 27th European Signal Processing Conference (EUSIPCO), A Coruna, Spain, 2019, pp. 1–5, doi: 10.23919/EUSIPCO.2019.8902578.
