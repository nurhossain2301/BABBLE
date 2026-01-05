import torch
from torch import nn
import torch.nn.functional as F

class DNN_Block(nn.Module):
    """Block for linear layers.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    neurons : int
        Size of the linear layers.
    activation : torch.nn.Module class
        Class definition to use for constructing activation layers.
    dropout : float
        Rate to use for dropping neurons.

    Example
    -------
    >>> inputs = torch.rand(10, 15, 128)
    >>> block = DNN_Block(input_shape=inputs.shape, neurons=64)
    >>> outputs = block(inputs)
    >>> outputs.shape
    torch.Size([10, 15, 64])
    """

    def __init__(
        self, input_shape, neurons, activation=torch.nn.LeakyReLU, dropout=0.15
    ):
        super(DNN_Block, self).__init__()
        self.dnn_block = nn.Sequential(
            nn.Linear(input_shape, neurons),  # First linear layer
            # torch.nn.BatchNorm1d(neurons),                          
            activation(),                       # Activation function
            torch.nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.dnn_block(x)
    

    # class EncoderTrainer():
        