# %% =======================
# AE inference-only: encode full CachedGaussianDataset to embeddings (raw_idx aligned)
# =========================

import os
import time
import torch
from torch.utils.data import DataLoader

from util.seed_everything import seed_everything
from dataset.tahoe100m.log1p.gaussian.cached import CachedGaussianDataset
from mlp_mape_and_mse import MLPAutoencoder  # 你的模型定义

seed_everything(42)

# =========================
# Config
# =========================
DEVICE = torch.device("cuda:0")  # or torch.device("cpu")
BATCH_SIZE = 128
NUM_WORKERS = 4

# AE structure (必须和训练时一致，否则load会报错或效果不对)
LATENT_DIM = 1024
ENC_HIDDEN = [4096, 2048, 1024]
DEC_HIDDEN = [1024, 2048, 4096]
RESIDUAL_BLOCK_CNT = 0

# 你要加载的权重（改成你的best权重路径）
WEIGHT_PATH = "/ml_storage/jiaxin/catch_up/best_performance_ae/lfc_trying/ae_weight/1773245719.627751_best.pth"

OUT_DIR = "/ml_storage/jiaxin/catch_up/best_performance_ae/lfc_trying/"
ENCODED_OUT_DIR = os.path.join(OUT_DIR, "embedding/encoded")
os.makedirs(ENCODED_OUT_DIR, exist_ok=True)


# =========================
# Dataset transform
# =========================
def encode_log1p_mean(data, idx):
    return torch.tensor(data["log1p_mean"], dtype=torch.float32)


# =========================
# Main
# =========================
if __name__ == "__main__":
    # 1) build full dataset (raw order)
    dataset_all = CachedGaussianDataset(transform=encode_log1p_mean)
    input_dim = int(dataset_all[0].shape[0])

    loader_all = DataLoader(
        dataset_all,
        batch_size=BATCH_SIZE,
        shuffle=False,            # ✅ 关键：保证输出embedding顺序=raw_idx顺序
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    # 2) init model + load weights
    model = MLPAutoencoder(
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        encoder_hidden=ENC_HIDDEN,
        decoder_hidden=DEC_HIDDEN,
        residual_block_cnt=RESIDUAL_BLOCK_CNT,
    ).to(DEVICE)

    state = torch.load(WEIGHT_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    # 3) encode
    embeddings = []
    with torch.inference_mode():  # 比 no_grad 更适合纯推理
        for batch in loader_all:
            batch = batch.to(DEVICE, non_blocking=True)
            z = model.encode(batch)         # [B, LATENT_DIM]
            embeddings.append(z.cpu())

    embeddings = torch.cat(embeddings, dim=0)  # [N, LATENT_DIM]
    print("embeddings.shape =", tuple(embeddings.shape))
    print("aligned: embeddings[i] <-> dataset_all[i] (raw_idx=i)")

    # 4) save
    time_tag = time.time()
    out_path = os.path.join(ENCODED_OUT_DIR, f"{time_tag}.pt")
    torch.save(embeddings, out_path)
    print(f"✅ saved encoded embeddings: {out_path}")
