from torch import nn
import torch


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM model
    Reference:
      Xingjian Shi et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting."
    """

    def __init__(self, input_channel, hidden_size, kernel_size, stride=1, padding=1, bias=True, activation=torch.sigmoid):
        super().__init__()
        self._input_channel = input_channel
        self._kernel_size = kernel_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._conv = nn.Conv2d(in_channels=self._input_channel + self._hidden_size,
                               ###hidden state has similar spational struture as inputs, we simply concatenate them on the feature dimension
                               out_channels=self._hidden_size*4,  ##lstm has four gates
                               kernel_size=self._kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=bias)

    # x: Batch * input_channel * H * W
    # _hidden, _cell: Batch * hidden_size * H * W
    def forward(self, x, state):
        _hidden, _cell = state
        cat_x = torch.cat([x, _hidden], dim=1)
        Conv_x = self._conv(cat_x)

        i, f, o, j = torch.chunk(Conv_x, 4, dim=1)

        i = self._activation(i)
        f = self._activation(f)
        cell = _cell * f + i * torch.tanh(j)
        o = self._activation(o)
        hidden = o * torch.tanh(cell)

        return o, (hidden, cell)

class ConvLSTM(nn.Module):

    def __init__(self, input_channel, hidden_size, kernel_size, stride=1, padding=1, bias=True, activation=torch.sigmoid):
        super().__init__()
        self._cell = ConvLSTMCell(input_channel, hidden_size, kernel_size, stride=stride, padding=padding, bias=bias, activation=activation)

    # input: Batch * Seq * input_channel * H * W
    # state:
        # if no input: Zeros
        # else last hidden state
    def forward(self, input, state):
        seq_size, batch_size, _, height, width = input.size()
        # hidden = torch.zeros((batch_size, self._cell._hidden_size, height, width), dtype=input.dtype)
        # cell = torch.zeros((batch_size, self._cell._hidden_size, height, width), dtype=input.dtype)
        # state = (hidden, cell)
        outputs = []
        for i in range(seq_size):
            output, state = self._cell(input[i, ...], state)
            outputs.append(output)
        return torch.stack(outputs), state
