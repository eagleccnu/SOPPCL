import time
import dill
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from modules import VAE, DataAugmentation, nt_xent_loss


def load_contrastive_dataset(path_26: str, path_2755: str) -> torch.Tensor:
    with open(path_26, "rb") as f:
        array_data_26 = dill.load(f)
    with open(path_2755, "rb") as f:
        array_data_2755 = dill.load(f)

    all_array_data = np.concatenate((array_data_26, array_data_2755[:, :-1]), axis=0)
    data = all_array_data.astype(np.float32)
    return torch.tensor(data)


def train_contrastive_vae(
    data_tensor: torch.Tensor,
    input_dim: int = 1310,
    hidden_dim: int = 256,
    latent_dim: int = 64,
    batch_size: int = 128,
    lr: float = 1e-3,
    num_epochs: int = 100,
    patience: int = 10,
    save_path: str = "contrastive_model.pth",
    device: torch.device | None = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    augment = DataAugmentation()

    best_loss = float("inf")
    counter = 0
    train_losses: list[float] = []

    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0.0
        model.train()

        for (x,) in dataloader:
            x = x.to(device)

            x1, x2 = augment(x)
            z1 = model(x1)
            z2 = model(x2)

            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        train_losses.append(epoch_loss)
        elapsed = int(time.time() - start_time)

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"| Time: {elapsed}s - Train Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered.")
            break

    # 绘制 loss 曲线
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = load_contrastive_dataset(
        "esm_hmm_label_26.dat",
        "esm_hmm_label_2755.dat",
    )
    train_contrastive_vae(
        data_tensor=features,
        device=device,
        save_path="contrastive_model.pth",
    )
