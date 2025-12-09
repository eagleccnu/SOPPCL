import torch
import torch.nn as nn
import torch.nn.functional as F


class DataAugmentation:

    def __init__(self, noise_std: float = 0.1, mask_prob: float = 0.1):
        self.noise_std = noise_std
        self.mask_prob = mask_prob

    def __call__(self, x: torch.Tensor):
        noise = torch.randn_like(x) * self.noise_std
        x_noisy = x + noise

        mask = torch.rand_like(x) < self.mask_prob
        x_masked = x.clone()
        x_masked[mask] = 0.0

        return x_noisy, x_masked


class Encoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor):
        h = torch.relu(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(h))
        return x_recon


class VAE(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 1.0):

    batch_size = z_i.size(0)

    z = torch.cat([z_i, z_j], dim=0)
    similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

    sim_ij = torch.diag(similarity, batch_size)
    sim_ji = torch.diag(similarity, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    mask = (~torch.eye(batch_size * 2, batch_size * 2,
                       dtype=torch.bool, device=z_i.device)).float()
    numerator = torch.exp(positives / temperature)
    denominator = mask * torch.exp(similarity / temperature)

    all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
    loss = torch.sum(all_losses) / (2 * batch_size)
    return loss


class BiLSTM(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=0.2
        )

        self.fc1 = nn.Linear(2 * hidden_dim, 1024)
        # self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)
        out = torch.relu(self.fc1(out))
        # out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        return out.squeeze(dim=-1)

