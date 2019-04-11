from torch import nn
import torch
from nowcasting.utils import make_layers

class Forecaster(nn.Module):
    def __init__(self, subnets, rnns, RNN):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, params in enumerate(subnets):
            setattr(self, 'stage' + str(self.blocks-index), make_layers(params))

        for index, params in enumerate(rnns):
            setattr(self, 'rnn' + str(self.blocks-index), RNN(params[0], params[0], params[0]))

    def forward_by_stage(self, input, state, subnet, rnn):
        input, state_stage = rnn(input, state)
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))

        return input, state_stage

        # input: 5D S*B*I*H*W

    def forward(self, hidden_states, size):
        input = torch.zeros(size, dtype=hidden_states.dtype)
        for i in list(range(1, self.blocks + 1))[::-1]:
            input, state_stage = self.forward_by_stage(input, hidden_states[i-1], getattr(self, 'stage' + str(i)),
                                                       getattr(self, 'rnn' + str(i)))
        return input
