from torch import nn


class EF(nn.Module):

    def __init__(self, encoder, forecaster):
        super().__init__()
        self.encoder = encoder
        self.forecaster = forecaster

    def forward(self, input):
        state, size = self.encoder(input)
        output = self.forecaster(state, size)
        return output