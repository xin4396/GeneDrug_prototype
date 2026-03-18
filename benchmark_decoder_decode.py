# decode pred result through best decoder to get gene expression for benchmark
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.seed_everything import seed_everything

seed_everything(42)

from mlp_mape_and_mse import MLPAutoencoder

# %%
import os
import torch

# from util.dataset import dataset, cell_line_to_id, drug_name_to_id
from dataset.tahoe100m.log1p.gaussian.cached import CachedGaussianDataset
from torch.utils.data import DataLoader, random_split


def encode_data_point(data, idx):
    # sigma = torch.tensor(data["sigma"], dtype=torch.float32)
    log1p_mean = torch.tensor(data["log1p_mean"], dtype=torch.float32)
    # logvar = torch.log(sigma + 1e-6) * 2
    # x = torch.cat([mu, logvar], dim=-1)
    # drug_name = data["drug_name"]
    # drug_label = drug_name_to_id[drug_name]
    # drug_conc = torch.tensor(data["drug_conc"], dtype=torch.float32)
    # log1p_drug_conc = torch.log1p(drug_conc)
    return log1p_mean


# %%
import time
from torch.utils.tensorboard import SummaryWriter

# %%
if __name__ == "__main__":

    # %%
    device = torch.device("cuda:0")

    input_dim = 62710
    model = MLPAutoencoder(
        input_dim=input_dim,
        latent_dim=1024,
        encoder_hidden=[4096, 2048, 1024],
        decoder_hidden=[1024, 2048, 4096],
        residual_block_cnt=0,
    ).to(device)
    model.load_state_dict(
        torch.load("/ml_storage/jiaxin/catch_up/best_performance_ae/lfc_trying/ae_weight/1773245719.627751_best.pth")
    )

    ## now encode the entire dataset
    from torch.utils.data import DataLoader, random_split
    # %%
    ## decode from predicted embeddings
    import numpy as np

    #load embeddings from pred results dir 
    with np.load(
        "/ml_storage/jiaxin/catch_up/best_performance_ae/lfc_trying/embedding/mlp_raw_predict/residual_swiglu.npz"
    ) as data:
        predicted_gene_embs = data["predicted_gene_embs"]
    len(predicted_gene_embs)
    # %%
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(predicted_gene_embs, dtype=torch.float32)
    )
    loader = DataLoader(dataset, num_workers=4, batch_size=128, persistent_workers=True)

    print(model.training)          # False
    print(model.encoder.training)  # False（如果 encoder 是 nn.Module 子模块）
    # %%
    model.eval()
    print(model.training)          # False
    print(model.encoder.training)  # False（如果 encoder 是 nn.Module 子模块）
    decoded_means = []
    with torch.no_grad():
        for batch in loader:
            batch = batch[0].to(device)
            _, mean_mask, mean = model.decode(batch)
            decoded_means.append((mean * (mean_mask > 0.5)).cpu().numpy())
    decoded_means = np.concatenate(decoded_means, axis=0)

    np.savez_compressed(
        f"/ml_storage/jiaxin/catch_up/best_performance_ae/lfc_trying/embedding/mlp_predict_decoded/downstream_residual_swiglu_decoded.npz",
        decoded_means=decoded_means,
    )
