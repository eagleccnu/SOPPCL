import time
import dill
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Dataset

from modules import VAE, DataAugmentation, BiLSTM


torch.manual_seed(123)
torch.cuda.manual_seed_all(123)


class OrderParameterDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


def load_supervised_dataset(path_26: str, path_2755: str):
    with open(path_26, "rb") as f:
        array_data_26 = dill.load(f)
    with open(path_2755, "rb") as f:
        array_data_2755 = dill.load(f)
    all_array_data = np.concatenate((array_data_26, array_data_2755), axis=0)

    np.random.seed(100)
    np.random.shuffle(all_array_data)

    features = all_array_data[:, :-1].astype(np.float32)
    labels = all_array_data[:, -1].astype(np.float32)

    return torch.from_numpy(features), torch.from_numpy(labels)


def augment_features_with_contrastive_vae(
    features: torch.Tensor,
    vae_weight_path: str,
    input_dim: int = 1310,
    hidden_dim: int = 256,
    latent_dim: int = 64,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = VAE(input_dim, hidden_dim, latent_dim).to(device)
    vae.load_state_dict(torch.load(vae_weight_path, map_location=device))
    vae.eval()

    augment = DataAugmentation()

    with torch.no_grad():
        x = features.to(device)
        x_noisy, x_masked = augment(x)
        feat_aug_1 = vae(x_noisy)
        feat_aug_2 = vae(x_masked)

    feats_cat = torch.cat([features, feat_aug_1.cpu(), feat_aug_2.cpu()], dim=0)
    return feats_cat


def train_regression_bilstm(
    features: torch.Tensor,
    labels: torch.Tensor,
    input_dim: int = 1310,
    hidden_dim: int = 1024,
    num_layers: int = 2,
    batch_size: int = 1024,
    lr: float = 1e-5,
    num_epochs: int = 200,
    patience: int = 10,
    save_path: str = "regression_model.pth",
    device: torch.device | None = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_np = labels.numpy()
    labels_cat = np.concatenate([labels_np, labels_np, labels_np], axis=0)
    labels_cat = torch.from_numpy(labels_cat).float()

    features = features.unsqueeze(1)
    labels_cat = labels_cat.unsqueeze(1)

    dataset = OrderParameterDataset(features, labels_cat)
    length = len(dataset)
    train_size = int(0.9 * length)
    val_size = length - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BiLSTM(input_dim, hidden_dim, num_layers).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    counter = 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        total_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device).float()
                y_batch = y_batch.to(device).float()
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        elapsed = int(time.time() - start_time)
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Time: {elapsed}s "
              f"- Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered.")
            break

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feats, labels = load_supervised_dataset(
        "esm_hmm_label_26.dat",
        "esm_hmm_label_2755.dat",
    )

    feats_aug = augment_features_with_contrastive_vae(
        feats, "contrastive_model.pth", device=device
    )

    train_regression_bilstm(
        features=feats_aug,
        labels=labels,
        device=device,
        save_path="regression_model.pth",
    )
