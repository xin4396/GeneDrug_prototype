# %% =======================
#  AE training with split derived from PairedGaussianDataset.paired_indices
#  - build raw_idx split from paired_idx split
#  - train AE only on train_raw (no val/test leakage)
#  - encode full raw dataset to keep index-alignment for step2 (gene_embs[index])
# =========================

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from util.seed_everything import seed_everything
from dataset.tahoe100m.log1p.gaussian.cached import CachedGaussianDataset

seed_everything(42)

# =========================
# Config
# =========================
DRUG_CTRL_NAME = "DMSO_TF"

PAIRED_SPLIT_PATH = "/home/mark/repos/genedrug/cache/dataset/tahoe100m/log1p/gaussian/paired_indices_split.np"

OUT_DIR = "/home/clab/Downloads/jiaxin_temporal/catch_up/best_performance_ae/relmape/"
ENCODED_OUT_DIR = os.path.join(OUT_DIR, "embedding/encoded")
WEIGHT_OUT_DIR = os.path.join(OUT_DIR, "ae_weight")

os.makedirs(WEIGHT_OUT_DIR, exist_ok=True)
os.makedirs(ENCODED_OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0")
BATCH_SIZE = 128
NUM_WORKERS = 4

# AE structure
LATENT_DIM = 1024
ENC_HIDDEN = [4096, 2048, 1024]
DEC_HIDDEN = [1024, 2048, 4096]
RESIDUAL_BLOCK_CNT = 0

# training
NUM_EPOCHS = 10000
LR = 1e-4
WEIGHT_DECAY = 1e-5

# IMPORTANT: whether include ctrl samples into AE datasets
INCLUDE_CTRL_IN_AE = True

# If True, force raw splits disjoint (recommended to avoid leakage via shared ctrl)
FORCE_DISJOINT_RAW_SPLITS = True



class ResidualBlock(nn.Module):
    """Residual block for MLP with pre-activation structure"""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim * 2)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x * 0.3  # Scale residual for stability


class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim=62710,
        latent_dim=512,
        hidden_dims=[2048, 1024, 768],
        residual_block_cnt=1,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, latent_dim))
        for _ in range(residual_block_cnt):
            layers.append(ResidualBlock(latent_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class DecoderHeads(nn.Module):
    """
    Return:
      masked_mean : sigmoid(mask_logits) * mean
      mask_logits : logits (for BCEWithLogits)
      mean        : raw mean
    """
    def __init__(self, prev_dim, output_dim):
        super().__init__()
        self.mask_layer = nn.Linear(prev_dim, output_dim)
        self.mean_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        mask_logits = self.mask_layer(x)
        mean = self.mean_layer(x)
        mask_prob = torch.sigmoid(mask_logits)
        masked_mean = mask_prob * mean
        return masked_mean, mask_logits, mean


class MLPDecoder(nn.Module):
    def __init__(
        self,
        latent_dim=512,
        output_dim=62710,
        hidden_dims=[768, 1024, 2048],
        residual_block_cnt=1,
    ):
        super().__init__()
        layers = []
        prev_dim = latent_dim

        for _ in range(residual_block_cnt):
            layers.append(ResidualBlock(latent_dim))

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        self.decoder_trunk = nn.Sequential(*layers)
        self.decoder_heads = DecoderHeads(prev_dim=prev_dim, output_dim=output_dim)

    def forward(self, z):
        h = self.decoder_trunk(z)
        return self.decoder_heads(h)


class MLPAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim=62710,
        latent_dim=512,
        encoder_hidden=None,
        decoder_hidden=None,
        residual_block_cnt=1,
    ):
        super().__init__()
        if encoder_hidden is None:
            encoder_hidden = [2048, 1024, 768]
        if decoder_hidden is None:
            decoder_hidden = [768, 1024, 2048]

        self.encoder = MLPEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden,
            residual_block_cnt=residual_block_cnt,
        )
        self.decoder = MLPDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=decoder_hidden,
            residual_block_cnt=residual_block_cnt,
        )

    def forward(self, x):
        z = self.encoder(x)
        masked_mean, mask_logits, mean = self.decoder(z)
        return masked_mean, mask_logits, mean

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# =========================
# Loss
# =========================
def multihead_loss(mask_logits, mean_pred, mean_true, alpha=0.6):
    """
    mask_logits: logits (NOT sigmoid prob)
    mean_pred  : predicted mean (raw, not masked)
    mean_true  : target vector
    """
    mean_mask_true = (mean_true != 0).float()

    pos_weight = torch.tensor([(1 - alpha) / alpha], device=mask_logits.device)
    loss_mask = F.binary_cross_entropy_with_logits(
        mask_logits, mean_mask_true, pos_weight=pos_weight
    )

    non_zero_idx = mean_mask_true.bool()
    if non_zero_idx.any():
        loss_mean = F.mse_loss(mean_pred[non_zero_idx], mean_true[non_zero_idx])
    else:
        loss_mean = torch.tensor(0.0, device=mean_pred.device)

    return loss_mask + loss_mean, loss_mask, loss_mean


# =========================
# Build paired_idx -> raw_idx mapping (lightweight, no 62710 vector loaded here)
# =========================
def encode_meta_only(data, index):
    plate = data["plate"]
    cell_line_str = data["cell_line"]
    drug_name = data["drug_name"].strip()
    return plate, cell_line_str, drug_name


class PairedIndexMapper:
    """
    Build:
      paired_indices[paired_idx] = (perturbed_raw_i, ctrl_raw_i)
    using the same logic as your PairedGaussianDataset, but meta-only for speed.
    """
    def __init__(self, drug_ctrl_name=DRUG_CTRL_NAME):
        self.drug_ctrl_name = drug_ctrl_name
        self.raw_dataset = CachedGaussianDataset(transform=encode_meta_only)

        # ctrl map: ctrl_map[plate][cell_line] = raw_idx (last wins)
        self.ctrl_map = {}
        for i in range(len(self.raw_dataset)):
            plate, cell, drug = self.raw_dataset[i]
            if drug == self.drug_ctrl_name:
                self.ctrl_map.setdefault(plate, {})
                self.ctrl_map[plate][cell] = i

        self.paired_indices = []
        for i in range(len(self.raw_dataset)):
            plate, cell, drug = self.raw_dataset[i]
            if drug != self.drug_ctrl_name:
                if plate in self.ctrl_map and cell in self.ctrl_map[plate]:
                    ctrl_i = self.ctrl_map[plate][cell]
                    self.paired_indices.append((i, ctrl_i))


def load_paired_split_indices(path=PAIRED_SPLIT_PATH):
    with open(path, "rb") as f:
        train_paired = np.load(f).tolist()
        val_paired = np.load(f).tolist()
        test_paired = np.load(f).tolist()
    return train_paired, val_paired, test_paired


def paired_to_raw_set(paired_indices, paired_idx_list, include_ctrl: bool):
    s = set()
    for pidx in paired_idx_list:
        pert_raw, ctrl_raw = paired_indices[pidx]
        s.add(int(pert_raw))
        if include_ctrl:
            s.add(int(ctrl_raw))
    return s


def make_raw_splits_from_paired(mapper: PairedIndexMapper,
                               include_ctrl_in_ae: bool = INCLUDE_CTRL_IN_AE,
                               force_disjoint: bool = FORCE_DISJOINT_RAW_SPLITS):
    train_paired, val_paired, test_paired = load_paired_split_indices()

    # sanity: split max index should be < len(paired_indices)
    max_idx = max(max(train_paired), max(val_paired), max(test_paired))
    if max_idx >= len(mapper.paired_indices):
        raise RuntimeError(
            f"paired_indices_split.np has paired_idx={max_idx} but mapper.paired_indices len={len(mapper.paired_indices)}.\n"
            f"Likely your mapper logic != the one used to create split."
        )

    train_raw = paired_to_raw_set(mapper.paired_indices, train_paired, include_ctrl_in_ae)
    val_raw = paired_to_raw_set(mapper.paired_indices, val_paired, include_ctrl_in_ae)
    test_raw = paired_to_raw_set(mapper.paired_indices, test_paired, include_ctrl_in_ae)

    if force_disjoint:
        # ensure no leakage: train > val > test priority
        val_before = len(val_raw)
        test_before = len(test_raw)
        val_raw = val_raw - train_raw
        test_raw = test_raw - train_raw - val_raw
        print(f"[disjoint] val removed {val_before - len(val_raw)} samples overlapping train")
        print(f"[disjoint] test removed {test_before - len(test_raw)} samples overlapping train/val")

    # sort for determinism
    train_raw = sorted(train_raw)
    val_raw = sorted(val_raw)
    test_raw = sorted(test_raw)

    print("==== Raw split stats (for AE) ====")
    print("include_ctrl_in_ae:", include_ctrl_in_ae, "| force_disjoint:", force_disjoint)
    print("train_raw:", len(train_raw), "val_raw:", len(val_raw), "test_raw:", len(test_raw))
    print("=================================")

    return train_raw, val_raw, test_raw


# =========================
# AE dataset: raw -> log1p_mean tensor
# =========================
def encode_log1p_mean(data, idx):
    return torch.tensor(data["log1p_mean"], dtype=torch.float32)


# =========================
# Main
# =========================
if __name__ == "__main__":
    # 1) build mapper + raw splits
    mapper = PairedIndexMapper(drug_ctrl_name=DRUG_CTRL_NAME)
    train_raw_idx, val_raw_idx, test_raw_idx = make_raw_splits_from_paired(
        mapper,
        include_ctrl_in_ae=INCLUDE_CTRL_IN_AE,
        force_disjoint=FORCE_DISJOINT_RAW_SPLITS,
    )

    # 2) build AE dataset (keeps raw order; Subset uses raw_idx)
    dataset_all = CachedGaussianDataset(transform=encode_log1p_mean)
    input_dim = int(dataset_all[0].shape[0])

    train_dataset = Subset(dataset_all, train_raw_idx)
    val_dataset = Subset(dataset_all, val_raw_idx)
    test_dataset = Subset(dataset_all, test_raw_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )

    # 3) init model
    model = MLPAutoencoder(
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        encoder_hidden=ENC_HIDDEN,
        decoder_hidden=DEC_HIDDEN,
        residual_block_cnt=RESIDUAL_BLOCK_CNT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    mse_loss_fn = nn.MSELoss()

    time_start = time.time()
    writer = SummaryWriter(f"/home/clab/Downloads/jiaxin_temporal/catch_up/best_performance_ae/run/ae/{time_start}")
    min_val_loss = float("inf")

    # =========================
    # 4) Train AE
    # =========================
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss_train = 0.0
        total_loss_mask_train = 0.0
        total_loss_mean_train = 0.0
        seen_train = 0

        for batch_mean_true in train_loader:
            batch_mean_true = batch_mean_true.to(DEVICE)
            bs = batch_mean_true.size(0)
            seen_train += bs

            optimizer.zero_grad()

            masked_pred, mask_logits, mean_pred = model(batch_mean_true)
            loss_total, loss_mask, loss_mean = multihead_loss(
                mask_logits=mask_logits,
                mean_pred=mean_pred,
                mean_true=batch_mean_true,
            )

            loss_total.backward()
            optimizer.step()

            total_loss_train += loss_total.item() * bs
            total_loss_mask_train += loss_mask.item() * bs
            total_loss_mean_train += loss_mean.item() * bs

        scheduler.step()

        # eval every epoch (you can change frequency)
        model.eval()
        total_loss_val = 0.0
        total_loss_mask_val = 0.0
        total_loss_mean_val = 0.0
        total_loss_val_mse = 0.0
        seen_val = 0

        with torch.no_grad():
            for batch_mean_true in val_loader:
                batch_mean_true = batch_mean_true.to(DEVICE)
                bs = batch_mean_true.size(0)
                seen_val += bs

                masked_pred, mask_logits, mean_pred = model(batch_mean_true)
                loss_total, loss_mask, loss_mean = multihead_loss(
                    mask_logits=mask_logits,
                    mean_pred=mean_pred,
                    mean_true=batch_mean_true,
                )

                total_loss_val += loss_total.item() * bs
                total_loss_mask_val += loss_mask.item() * bs
                total_loss_mean_val += loss_mean.item() * bs

                loss_mse = mse_loss_fn(masked_pred, batch_mean_true)
                total_loss_val_mse += loss_mse.item() * bs

        avg_loss_train = total_loss_train / max(seen_train, 1)
        avg_loss_mask_train = total_loss_mask_train / max(seen_train, 1)
        avg_loss_mean_train = total_loss_mean_train / max(seen_train, 1)

        avg_loss_val = total_loss_val / max(seen_val, 1)
        avg_loss_mask_val = total_loss_mask_val / max(seen_val, 1)
        avg_loss_mean_val = total_loss_mean_val / max(seen_val, 1)

        avg_loss_val_mse = total_loss_val_mse / max(seen_val, 1)

        print(
            f"epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"train: {avg_loss_train:.6f} (mask {avg_loss_mask_train:.6f}, mean {avg_loss_mean_train:.6f}) | "
            f"val: {avg_loss_val:.6f} (mask {avg_loss_mask_val:.6f}, mean {avg_loss_mean_val:.6f}) | "
            f"val_mse(masked): {avg_loss_val_mse:.6f}"
        )

        writer.add_scalar("loss/train_total", avg_loss_train, epoch)
        writer.add_scalar("loss/train_mask", avg_loss_mask_train, epoch)
        writer.add_scalar("loss/train_mean", avg_loss_mean_train, epoch)

        writer.add_scalar("loss/val_total", avg_loss_val, epoch)
        writer.add_scalar("loss/val_correct_mask_rate`", avg_loss_mask_val, epoch)
        writer.add_scalar("loss/val_mask_mse", avg_loss_mean_val, epoch) #依据true non-zero mask， 直接忽略0的位置
        writer.add_scalar("loss/val_mse_logit_masked_mse", avg_loss_val_mse, epoch)  #所有位置依据sigmoid（logits）

        # save best (keep your old heuristic if you want)
        if avg_loss_val < min_val_loss:
            min_val_loss = avg_loss_val
            weight_path = os.path.join(WEIGHT_OUT_DIR, f"{time_start}_best.pth")
            torch.save(model.state_dict(), weight_path)
            print(f"✅ saved best weight: {weight_path}")

    # =========================
    # 5) Encode full raw dataset (KEEP INDEX ALIGNMENT!)
    # =========================
    model.eval()
    loader_all = DataLoader(
        dataset_all,
        batch_size=BATCH_SIZE,
        shuffle=False,  # IMPORTANT: keep raw index order
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )

    embeddings = []
    with torch.no_grad():
        for batch in loader_all:
            batch = batch.to(DEVICE)
            z = model.encode(batch)
            embeddings.append(z.cpu())
    embeddings = torch.cat(embeddings, dim=0)

    encoded_path = os.path.join(ENCODED_OUT_DIR, f"{time_start}.pt")
    torch.save(embeddings, encoded_path)
    print(f"✅ saved encoded embeddings (aligned to raw idx): {encoded_path}")
    print("embeddings.shape =", tuple(embeddings.shape))

    # =========================
    # Optional: quick reconstruction demo
    # =========================
    print("---- quick recon demo ----")
    with torch.no_grad():
        for i in range(3):
            x = dataset_all[i].to(DEVICE)
            masked_pred, mask_logits, mean_pred = model(x.unsqueeze(0))
            mask_prob = torch.sigmoid(mask_logits)
            recon_masked = (mean_pred * (mask_prob > 0.5)).squeeze(0)
            print("Original     :", x[:8].detach().cpu())
            print("Recon_masked :", recon_masked[:8].detach().cpu())
            print("Recon_mean   :", mean_pred.squeeze(0)[:8].detach().cpu())
            print()

    # =========================
    # NOTE:
    # 之后第二步你继续：
    #   gene_embs = torch.load(encoded_path)
    #   gene_embs[index] 就仍然与 CachedGaussianDataset 的 raw index 对齐
    # =========================
