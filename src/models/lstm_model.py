import torch.nn as nn


class RNNGeneration(nn.Module):
    def __init__(self, config):
        super(RNNGeneration, self).__init__()
        self.embeddings_layer = nn.Embedding(config['dict_state'], 128)
        self.rnn = nn.LSTM(128, 256, 3, dropout=0.2)
        self.decoder = nn.Linear(5120, config['dict_state'])

    def forward(self, x):
        emb = self.embeddings_layer(x)
        output, _ = self.rnn(emb)

        output = output.reshape(output.shape[0], -1)
        output = self.decoder(output)
        return output
