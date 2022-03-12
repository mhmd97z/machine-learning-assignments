import torch
import torch.nn as nn


def build_mlp(input_size, output_size, n_layers, size, activation= nn.Tanh, output_activation=None):
    """
        #inputs:
            input_size: dimension of inputs.
            output_size: dimension of outputs.
            n_layers: number of the layers in the Sequential model.
            size: width of each hidden layer.(for the sake of simplicity, we take the width of all the hidden layers the same)
            activation: activation function used after each hidden layer.
            output_activation: activation function used in the last layer.

        #outputs:
            model: the implemented model.
    """
    # TODO: Sequentially append the layers to the list.
    # TODO: Use Xavier initialization for the weights.
    # Hint: Look at nn.Linear and nn.init.

    layers = []
    for i in range(n_layers):
        if i == 0:
            layers.append(nn.Linear(input_size, size))
            nn.init.xavier_uniform_(layers[-1].weight, gain=nn.init.calculate_gain('tanh'))
            layers.append(activation())
        elif i == (n_layers - 1):
            layers.append(nn.Linear(size,output_size))
            nn.init.xavier_uniform_(layers[-1].weight, gain=nn.init.calculate_gain('tanh'))
        else:
            layers.append(nn.Linear(size, size))
            nn.init.xavier_uniform_(layers[-1].weight, gain=nn.init.calculate_gain('tanh'))
            layers.append(activation())

    model = nn.Sequential(*layers)

    return model

############################################
############################################

device= torch.device("cuda")
dtype= torch.float32

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)
