"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: beat_tracking_tcn/models/tcn.py
Description: Implements a non-causal temporal convolutional network based on
             the model used in Bai et al 2018 [1]
References:
[1] Bai, S., Kolter, J.Z. and Koltun, V., 2018. An empirical evaluation of
    generic convolutional and recurrent networks for sequence modeling. arXiv
    preprint arXiv:1803.01271.
"""
import torch.nn as nn
from torch.nn.utils import weight_norm


class NonCausalTemporalLayer(nn.Module):
    """
    Implements a non-causal temporal block. Based off the model described in
    Bai et al 2018, but with the notable difference that the dilated temporal
    convolution is non-causal.

    Also implements a parallel residual convolution, allowing detail from the
    original signal to influence the output.
    """
    def __init__(
            self,
            inputs,
            outputs,
            dilation,
            kernel_size=5,
            stride=1,
            padding=4,
            dropout=0.1):
        """
        Construct an instance of NonCausalTemporalLayer

        Arguments:
            inputs {int} -- Input size in samples
            outputs {int} -- Output size in samples
            dilation {int} -- Size of dilation in samples

        Keyword Arguments:
            kernel_size {int} -- Size of convolution kernel (default: {5})
            stride {int} -- Size of convolution stride (default: {1})
            padding {int} -- How much padding to apply in total. Note, this is
                             halved and applied equally at each end of the
                             convolution to make the model non-causal.
                             (default: {4})
            dropout {float} -- The probability of dropping out a connection
                               during training. (default: {0.1})
        """
        super(NonCausalTemporalLayer, self).__init__()

        self.conv1 = nn.Conv1d(
                inputs,
                outputs,
                kernel_size,
                stride=stride,
                padding=int(padding / 2),
                dilation=dilation)
        self.conv1 = weight_norm(self.conv1)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
                inputs,
                outputs,
                kernel_size,
                stride=stride,
                padding=int(padding / 2),
                dilation=dilation)
        self.conv2 = weight_norm(self.conv2)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(inputs, outputs, 1)\
            if inputs != outputs else None
        self.elu3 = nn.ELU()

        self._initialise_weights(self.conv1, self.conv2, self.downsample)

    def forward(self, x):
        """
        Feed a tensor forward through the layer.

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
        y = self.conv2(x)
        y = self.elu2(y)
        y = self.dropout2(y)

        if self.downsample is not None:
            y = y + self.downsample(x)

        y = self.elu3(y)

        return y

    def _initialise_weights(self, *layers):
        for layer in layers:
            if layer is not None:
                layer.weight.data.normal_(0, 0.01)


class NonCausalTemporalConvolutionalNetwork(nn.Module):
    """
    Implements a non-causal temporal convolutional network. Based off the model
    described in Bai et al 2018. Initialises and forwards through a number of
    NonCausalTemporalLayer instances, depending on construction parameters.
    """
    def __init__(self, inputs, channels, kernel_size=5, dropout=0.1):
        """
        Construct a NonCausalTemporalConvolutionalNetwork.

        Arguments:
            inputs {int} -- Network input length.
            channels {list[int]} -- List containing number of channels each
                                    constituent temporal layer should have.

        Keyword Arguments:
            kernel_size {int} -- Size of dilated convolution kernels.
                                 (default: {5})
            dropout {float} -- The probability of dropping out a connection
                               during training. (default: {0.1})
        """
        super(NonCausalTemporalConvolutionalNetwork, self).__init__()

        self.layers = []
        n_levels = len(channels)

        for i in range(n_levels):
            dilation = 2 ** i

            n_channels_in = channels[i - 1] if i > 0 else inputs
            n_channels_out = channels[i]

            self.layers.append(
                NonCausalTemporalLayer(
                    n_channels_in,
                    n_channels_out,
                    dilation,
                    kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout
                )
            )
        
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        """
        Feed a tensor forward through the network.

        Arguments:
            x {torch.Tensor} -- A PyTorch tensor of size specified in the
                                constructor.

        Returns:
            torch.Tensor -- A PyTorch tensor of size determined by the final
                            temporal convolutional layer.
        """
        y = self.net(x)
        return y
