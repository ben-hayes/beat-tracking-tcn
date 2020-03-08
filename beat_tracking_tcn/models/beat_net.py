"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: models/beat_net.py
Description: A CNN including a Temporal Convolutional Layer designed to predict
             a vector of beat activations from an input spectrogram (tested
             with mel spectrograms).
"""
import torch.nn as nn

from beat_tracking_tcn.models.tcn import NonCausalTemporalConvolutionalNetwork


class BeatNet(nn.Module):
    """
    PyTorch implementation of a BeatNet CNN. The network takes a
    mel-spectrogram input. It then learns an intermediate convolutional
    representation, and finally applies a non-causal Temporal Convolutional
    Network to predict a beat activation vector.

    The structure of this network is based on the model proposed in Davies &
    Bock 2019.
    """

    def __init__(
            self,
            input=(3000, 81),
            output=3000,
            channels=16,
            tcn_kernel_size=5,
            dropout=0.1,
            downbeats=False):
        """
        Construct an instance of BeatNet.

        Keyword Arguments:
            input {tuple} -- Input dimensions (default: {(3000, 81)})
            output {int} -- Output dimensions (default: {3000})
            channels {int} -- Convolution channels (default: {16})
            tcn_kernel_size {int} -- Size of dilated convolution kernels.
                                     (default: {5})
            dropout {float} -- Network connection dropout probability.
                               (default: {0.1})
        """
        super(BeatNet, self).__init__()

        self.conv1 = nn.Conv2d(1, channels, (3, 3), padding=(1, 0))
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool2d((1, 3))

        self.conv2 = nn.Conv2d(channels, channels, (3, 3), padding=(1, 0))
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)
        self.pool2 = nn.MaxPool2d((1, 3))

        self.conv3 = nn.Conv2d(channels, channels, (1, 8))
        self.elu3 = nn.ELU()
        self.dropout3 = nn.Dropout(dropout)

        self.tcn = NonCausalTemporalConvolutionalNetwork(
            channels,
            [channels] * 11,
            tcn_kernel_size,
            dropout)

        self.out = nn.Conv1d(16, 1 if not downbeats else 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Feed a tensor forward through the BeatNet.

        Arguments:
            x {torch.Tensor} -- A PyTorch tensor of size specified in the
                                constructor.

        Returns:
            torch.Tensor -- A PyTorch tensor of size specified in the
                            constructor.
        """
        y = self.conv1(x)
        y = self.elu1(y)
        y = self.dropout1(y)
        y = self.pool1(y)

        y = self.conv2(y)
        y = self.elu2(y)
        y = self.dropout2(y)
        y = self.pool2(y)

        y = self.conv3(y)
        y = self.elu3(y)

        y = y.view(-1, y.shape[1], y.shape[2])
        y = self.tcn(y)

        y = self.out(y)
        y = self.sigmoid(y)

        return y
