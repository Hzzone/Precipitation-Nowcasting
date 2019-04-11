from torch import nn
import torch
from nowcasting.utils import make_layers

class Encoder(nn.Module):
    def __init__(self, subnets, rnns, RNN):
        super().__init__()
        assert len(subnets)==len(rnns)

        self.blocks = len(subnets)

        for index, params in enumerate(subnets, 1):
            setattr(self, 'stage'+str(index), make_layers(params))

        for index, params in enumerate(rnns, 1):
            setattr(self, 'rnn'+str(index), RNN(params[0], params[0], params[0]))

    def forward_by_stage(self, input, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))
        hidden = torch.zeros((batch_size, rnn._cell._hidden_size, input.size(3), input.size(4)), dtype=input.dtype)
        cell = torch.zeros((batch_size, rnn._cell._hidden_size, input.size(3), input.size(4)), dtype=input.dtype)
        state = (hidden, cell)
        outputs_stage, state_stage = rnn(input, state)
        print(outputs_stage.size())

        return outputs_stage, state_stage

    # input: 5D S*B*I*H*W
    def forward(self, input):
        hidden_states = []
        for i in range(1, self.blocks+1):
            input, state_stage = self.forward_by_stage(input, getattr(self, 'stage'+str(i)), getattr(self, 'rnn'+str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states), input.size()

