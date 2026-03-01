import torch
import torch.nn as nn

class SLGRAE(nn.Module):
    def __init__(self, input_dim):
        super(SLGRAE, self).__init__()
        # Encoder: Mengompresi 44 fitur jadi representasi laten (16 fitur)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        # Decoder: Rekonstruksi balik ke 44 fitur
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid() 
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed