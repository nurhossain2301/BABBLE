import torch
from torch import nn
import torch.nn.functional as F


# def get_linear_layer(n_neurons, input_size, finetune_encoder=False, name="weighted_average"):
#     if name=="weighted_average":
#         return WeightedAverage(n_neurons, input_size=input_size, freeze=not finetune_encoder)
#     if name=="dnn_block":
#         return DNN_Block(input_size, n_neurons)
#     return None

def get_linear_layer(n_neurons, input_size, name="weighted_average", freeze=False):
    if name=="weighted_average":
        return WeightedAverage(n_neurons, input_size=input_size, freeze=freeze)
    if name=="dnn_block":
        return DNN_Block(input_size, n_neurons)
    return None

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

class WeightedAverage(torch.nn.Module):
    """Computes a linear transformation y = wx + b.

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_shape: tuple
        It is the shape of the input tensor.
    input_size: int
        Size of the input tensor.
    bias : bool
        If True, the additive bias b is adopted.
    combine_dims : bool
        If True and the input is 4D, combine 3rd and 4th dimensions of input.

    Example
    -------
    >>> inputs = torch.rand(10, 50, 40)
    >>> lin_t = Linear(input_shape=(10, 50, 40), n_neurons=100)
    >>> output = lin_t(inputs)
    >>> output.shape
    torch.Size([10, 50, 100])
    """

    def __init__(
        self,
        n_neurons,
        input_shape=None,
        input_size=None,
        freeze=False
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]

        # Weights are initialized following pytorch approach
        self.w = nn.Linear(input_size, n_neurons, bias=False)
        torch.nn.init.constant_(self.w.weight, 1/input_size)
        self.w.weight.requires_grad=not freeze

    def forward(self, x):
        # Apply softmax to weights to ensure they sum to 1
        normalized_weights = F.softmax(self.w.weight, dim=1)
        normalized_weights = torch.squeeze(normalized_weights)
        # Compute weighted average
        weighted_average = torch.matmul(x, normalized_weights)
        return weighted_average