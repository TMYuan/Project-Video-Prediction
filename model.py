import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable

use_gpu = torch.cuda.is_available()

class Encoder(nn.Module):
    """
    Encoder of LSTM VAE
    """
    def __init__(self, lstm_layer, seq_len, input_dim, hidden_dim, batch_size):
        super(Encoder, self).__init__()
        
        # Model Variable
        self.lstm_enc = nn.LSTM(input_dim, hidden_dim, lstm_layer, batch_first=True, bidirectional=False)
        self.fc_mu = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.fc_logvar = nn.Linear(hidden_dim, int(hidden_dim / 2))
        
        # Activation function Variable
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        # Hidden Variable
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

    def encode(self, x):
        # Return last sequence of LSTM as output
        h1, _ = self.lstm_enc(x)
        h1 = self.tanh(h1)
        return self.fc_mu(h1[:, -1, :]), self.fc_logvar(h1[:, -1, :])

    def reparameterize(self, mu, logvar):
        # Reparameterize trick
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            if use_gpu:
                eps = eps.cuda()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return mu, z
    
    
class Decoder(nn.Module):
    """
    Decoder of LSTM VAE
    """
    def __init__(self, lstm_layer, seq_len, input_dim, hidden_dim, batch_size):
        super(Decoder, self).__init__()
        
        # Model Variable
        self.lstm_dec = nn.LSTM(int(hidden_dim / 2), input_dim, lstm_layer, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(input_dim, input_dim)
        
        # Activation function Variable
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        # Hidden Variable
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

    def decode(self, z):
        # Create a fake dimension and expand in that dimensionto make conditional LSTM.
        z = z.view(z.shape[0], 1, -1)
        z = z.expand(-1, self.seq_len, -1)
        out, _ = self.lstm_dec(z)
        out = self.fc(out)
#         out = self.sigmoid(out)
        return out
    
    def forward(self, x):
        out = self.decode(x)
        return out

    
class LSTMVAE(nn.Module):
    def __init__(self, lstm_layer = 1, seq_len = 10, input_dim = 1, hidden_dim = 2, batch_size = 1):
        super(LSTMVAE, self).__init__()
        self.encoder = Encoder(lstm_layer, seq_len, input_dim, hidden_dim, batch_size)
        self.decoder = Decoder(lstm_layer, seq_len, input_dim, hidden_dim, batch_size)

    def forward(self, x):
        mu, z = self.encoder(x)
        out = self.decoder(z)
        return out, mu, z