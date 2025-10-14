import os
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pacmap
from tqdm import tqdm
from collections import OrderedDict

from model import Res34SimSiam, Res34SimSiamNoise
from load_data import LoadData

# ================== Config ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

layer_keys = ["pool0", "pool1", "pool2", "pool3", "avgpool"]

colors = {
    "ECG": "#1f77b4",        # blue
    "PPG": "#ff7f0e",        # orange
    "PPG Peaks": "#2ca02c",  # green
    "ECG Peaks": "#d62728",  # red
    "AF": "#9467bd",         # purple
    "Non-AF": "#8c564b"      # brown/gray
}



# ================== Utils ==================
def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = OrderedDict({k.replace("module.", ""): v for k, v in checkpoint.items()})
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def get_activations(dataloader, model, modules, source="PPG"):
    activations, hooks = {}, {}
    all_labels = []

    def get_activation(name):
        def hook(_, __, output):
            data = output.detach().cpu().numpy()
            activations[name] = np.concatenate([activations[name], data], axis=0) if name in activations else data
        return hook

    # Register hooks for pooling layers
    for module in modules:
        for name, layer in module.named_children():
            if "pool" in name:
                hooks[name] = layer.register_forward_hook(get_activation(name))

    for ECG, PPG, target in tqdm(dataloader, desc=f"Extracting {source}"):
        PPG, ECG = PPG.to(device).float().unsqueeze(1), ECG.to(device).float().unsqueeze(1)

        all_labels.append(target.cpu().numpy())
        model(PPG if source == "PPG" else ECG)

    for h in hooks.values():
        h.remove()

    labels = np.concatenate(all_labels, axis=0)
    return activations, labels


def reduce_and_save(activations, save_dir="val_latent_activation_npys"):
    os.makedirs(save_dir, exist_ok=True)
    for key, act in activations.items():
        act = act.reshape(act.shape[0], -1)
        embedding = pacmap.PaCMAP(n_components=2, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
        X_transformed = embedding.fit_transform(act, init="pca")
        np.save(os.path.join(save_dir, f"activation_transformed_{key}.npy"), X_transformed)
        print(f"[Saved] {key} -> {X_transformed.shape}")


def plot_embeddings(X, labels, key, save_dir="latent_vizs", is_last=False):
    os.makedirs(save_dir, exist_ok=True)

    n = labels.shape[0]
    labels_full = np.tile(labels, 4)   # match PPG, ECG, PPG Peaks, ECG Peaks

    idx_ranges = {
        "PPG": (0, n),
        "ECG": (n, 2 * n),
        "PPG Peaks": (2 * n, 3 * n),
        "ECG Peaks": (3 * n, 4 * n)
    }

    if not is_last:
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, (start, end) in idx_ranges.items():
            ax.scatter(X[start:end, 0], X[start:end, 1],
                       s=2, alpha=0.4, label=name, color=colors.get(name, "gray"))
        ax.axis("off")
        ax.legend(numpoints=10, markerscale=0.7)

    else:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

        # Modality separation
        for name, (start, end) in idx_ranges.items():
            ax0.scatter(X[start:end, 0], X[start:end, 1],
                        s=2, alpha=0.4, label=name, color=colors.get(name, "gray"))
        ax0.axis("off")
        ax0.legend(numpoints=10, markerscale=0.7)

        # AF vs Non-AF
        for name, (start, end) in idx_ranges.items():
            ax1.scatter(X[start:end, 0][labels_full[start:end] == 1], 
                        X[start:end, 1][labels_full[start:end] == 1],
                        s=2, alpha=0.4, label=f"{name} AF" if name == "PPG" else "", color=colors["AF"])
            ax1.scatter(X[start:end, 0][labels_full[start:end] == 0], 
                        X[start:end, 1][labels_full[start:end] == 0],
                        s=2, alpha=0.4, label=f"{name} Non-AF" if name == "PPG" else "", color=colors["Non-AF"])
        ax1.axis("off")
        ax1.legend(numpoints=10, markerscale=0.7)

    plt.savefig(os.path.join(save_dir, f"simsiam_{key}.png"), dpi=300)
    plt.close()


# ================== Main ==================
if __name__ == "__main__":
    # Paths
    val_data_path = "MIMIC_val.npz"
    val_data_peak_path = "MIMIC_val_peaks.npz"
    checkpoint_path = "saved_models/model_59.pt"
    batch_size = 64

    # Load Dataset
    val_loader = DataLoader(LoadData(val_data_path), batch_size=batch_size, shuffle=False)
    val_loader_peak = DataLoader(LoadData(val_data_peak_path), batch_size=batch_size, shuffle=False)

    # Load Model
    model = Res34SimSiamNoise(512, 128, predictor=True, single_source_mode=True).to(device)
    model = load_checkpoint(model, checkpoint_path)

    # Extract Activations
    PPG_acts, labels = get_activations(val_loader, model, [model.encoder], source="PPG")
    ECG_acts, _ = get_activations(val_loader, model, [model.encoder], source="ECG")
    PPG_peak_acts, _ = get_activations(val_loader_peak, model, [model.encoder], source="PPG")
    ECG_peak_acts, _ = get_activations(val_loader_peak, model, [model.encoder], source="ECG")

    # Combine & Reduce
    combined_acts = {
        key: np.concatenate([PPG_acts[key], ECG_acts[key], PPG_peak_acts[key], ECG_peak_acts[key]], axis=0)
        for key in PPG_acts.keys()
    }
    reduce_and_save(combined_acts)

    # Visualization
    for idx, key in enumerate(layer_keys):
        X_transformed = np.load(f"val_latent_activation_npys/activation_transformed_{key}.npy")
        plot_embeddings(X_transformed, labels, key, is_last=(idx == len(layer_keys) - 1))

