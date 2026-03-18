"""
Perturbation MLP training with fixed split (paired_indices_split.np)

Train/Val are strictly from:
  /home/mark/repos/genedrug/cache/dataset/tahoe100m/log1p/gaussian/paired_indices_split.np

Test is NOT used in training/early-stopping.
"""

import json
import os
from typing import Literal

import accelerate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.tahoe100m.h5ad import MultiH5ADDataset
from dataset.tahoe100m.log1p.gaussian.cached import CachedGaussianDataset


# --------------------- Model: LayerNorm + Residual + SwiGLU/GEGLU ---------------------
class GatedFFNBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_mult: int = 4,
        dropout: float = 0.1,
        gated_act: Literal["swiglu", "geglu"] = "swiglu",
        ln_eps: float = 1e-5,
        zero_init_out: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.hidden = dim * hidden_mult
        self.gated_act = gated_act

        self.ln = nn.LayerNorm(dim, eps=ln_eps)
        self.fc_in = nn.Linear(dim, 2 * self.hidden, bias=True)
        self.fc_out = nn.Linear(self.hidden, dim, bias=True)
        self.drop = nn.Dropout(dropout)

        if zero_init_out:
            nn.init.zeros_(self.fc_out.weight)
            nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc_in(h)
        v, g = h.chunk(2, dim=-1)

        if self.gated_act == "swiglu":
            h = v * F.silu(g)
        elif self.gated_act == "geglu":
            h = v * F.gelu(g)
        else:
            raise ValueError(f"Unknown gated_act: {self.gated_act}")

        h = self.fc_out(h)
        h = self.drop(h)
        return x + h


class ResidualGatedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        model_dim: int = 2048,
        n_blocks: int = 6,
        hidden_mult: int = 4,
        dropout: float = 0.1,
        gated_act: Literal["swiglu", "geglu"] = "swiglu",
        ln_eps: float = 1e-5,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, model_dim, bias=True),
            nn.LayerNorm(model_dim, eps=ln_eps),
        )

        self.blocks = nn.ModuleList(
            [
                GatedFFNBlock(
                    dim=model_dim,
                    hidden_mult=hidden_mult,
                    dropout=dropout,
                    gated_act=gated_act,
                    ln_eps=ln_eps,
                    zero_init_out=True,
                )
                for _ in range(n_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(model_dim, eps=ln_eps),
            nn.Linear(model_dim, output_dim, bias=True),
        )

        nn.init.normal_(self.head[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


# --------------------- Split utils ---------------------
def get_split_dataset_indices(
    split_path="/home/mark/repos/genedrug/cache/dataset/tahoe100m/log1p/gaussian/paired_indices_split.np",
):
    with open(split_path, "rb") as f:
        train_indices = np.load(f).tolist()
        val_indices = np.load(f).tolist()
        test_indices = np.load(f).tolist()
    return train_indices, val_indices, test_indices


def get_split_datasets(dataset, split_path=None):
    if split_path is None:
        train_indices, val_indices, test_indices = get_split_dataset_indices()
    else:
        train_indices, val_indices, test_indices = get_split_dataset_indices(split_path)

    n = len(dataset)
    if max(train_indices + val_indices + test_indices) >= n:
        raise RuntimeError(
            f"Split indices out of range: max={max(train_indices + val_indices + test_indices)} >= len(dataset)={n}.\n"
            f"这通常表示：你当前构建的 paired_indices 顺序，与当初生成 paired_indices_split.np 的顺序不一致。"
        )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    return train_dataset, val_dataset, test_dataset


# --------------------- Data / vocab utils ---------------------
def load_h5ad_paths():
    return [
        f"/ml_storage/ZMK/datasets/Tahoe100M_Original/plate{i}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
        for i in range(1, 15)
    ]


def build_cellline_and_drug_vocab(h5ad_paths):
    dataset_h5 = MultiH5ADDataset(
        h5ad_paths=h5ad_paths,
        component_keys=[
            ("obs", "cell_line"),
            ("obs", "drugname_drugconc"),
        ],
    )

    cell_lines = set()
    for i in range(len(dataset_h5.datasets)):
        cell_lines.update(
            dataset_h5.datasets[i].h5f["obs"]["cell_line"]["categories"][:].astype(str)
        )
    cell_line_to_id = {cl: i for i, cl in enumerate(sorted(cell_lines))}

    drug_names: set[str] = set()
    drugname_drugconc_bytes: set[bytes] = set()
    for i in range(len(dataset_h5.datasets)):
        drugname_drugconc_bytes.update(
            dataset_h5.datasets[i].h5f["obs"]["drugname_drugconc"]["categories"][:]
        )

    for s in drugname_drugconc_bytes:
        drug_name, drug_conc = eval(s.decode("utf-8"))[0][:2]
        drug_names.add(drug_name.strip())

    drug_name_to_id = {dn: i for i, dn in enumerate(sorted(drug_names))}
    return cell_line_to_id, drug_name_to_id


def load_drug_embeddings(csv_path):
    drug_emb_df = pd.read_csv(csv_path)
    drug_name_2_emb = {}

    for _, row in drug_emb_df.iterrows():
        dn = row["drug"].strip()
        emb = torch.tensor(json.loads(row["embedding"]), dtype=torch.float32)
        drug_name_2_emb[dn] = emb

    return drug_name_2_emb


def build_normalized_drug_emb_tensor(drug_name_to_id, drug_name_2_emb, device):
    drug_embs = [drug_name_2_emb[dn] for dn in sorted(drug_name_to_id.keys())]
    drug_embs_tensor = torch.stack(drug_embs)
    drug_embs_tensor = torch.nn.functional.normalize(drug_embs_tensor, p=2, dim=1).to(device)
    return drug_embs_tensor


def make_encode_raw_with_emb(gene_embs, drug_name_to_id, drug_name_2_emb):
    def encode_raw_with_emb(data, index):
        gene_emb = gene_embs[index]  # [1024], cpu tensor
        drug_name = data["drug_name"].strip()
        drug_id = drug_name_to_id[drug_name]
        drug_emb = drug_name_2_emb[drug_name]  # [512], cpu tensor
        drug_conc = torch.tensor(data["drug_conc"], dtype=torch.float32)
        log1p_drug_conc = torch.log1p(drug_conc)
        cell_line_str = data["cell_line"]
        plate = data["plate"]
        return plate, cell_line_str, drug_name, drug_id, gene_emb, drug_emb, log1p_drug_conc

    return encode_raw_with_emb


def encode_meta_only(data, index):
    plate = data["plate"]
    cell_line_str = data["cell_line"]
    drug_name = data["drug_name"].strip()
    drug_conc = float(data["drug_conc"])
    return plate, cell_line_str, drug_name, drug_conc


def build_drug_ctrl_idx(dataset_meta, drug_ctrl_name="DMSO_TF"):
    drug_ctrl_idx = {}
    for i in range(len(dataset_meta)):
        plate, cell_line_str, drug_name, drug_conc = dataset_meta[i]
        if drug_name == drug_ctrl_name:
            drug_ctrl_idx.setdefault(plate, {})
            drug_ctrl_idx[plate][cell_line_str] = i
    return drug_ctrl_idx


class PairedGaussianDataset(torch.utils.data.Dataset):
    """
    idx (paired_idx) -> (input_emb, target_emb, drug_name_p)
    where:
      input_emb  = [z_ctrl(1024) ; drug_emb(512) ; log1p_conc(1)]
      target_emb = z_pert(1024)
    """

    def __init__(self, raw_dataset, meta_dataset, drug_ctrl_idx, drug_ctrl_name="DMSO_TF"):
        self.raw_dataset = raw_dataset
        self.meta_dataset = meta_dataset
        self.drug_ctrl_idx = drug_ctrl_idx
        self.drug_ctrl_name = drug_ctrl_name

        self.paired_indices = []
        for i in range(len(meta_dataset)):
            plate, cell_line_str, drug_name, drug_conc = meta_dataset[i]
            if drug_name != self.drug_ctrl_name:
                if plate in drug_ctrl_idx and cell_line_str in drug_ctrl_idx[plate]:
                    ctrl_i = drug_ctrl_idx[plate][cell_line_str]
                    self.paired_indices.append((i, ctrl_i))

    def __len__(self):
        return len(self.paired_indices)

    def __getitem__(self, idx):
        pert_i, ctrl_i = self.paired_indices[idx]

        plate_p, cell_line_p, drug_name_p, drug_id_p, gene_emb_p, drug_emb_p, log1p_conc_p = self.raw_dataset[pert_i]
        plate_c, cell_line_c, drug_name_c, drug_id_c, gene_emb_c, drug_emb_c, log1p_conc_c = self.raw_dataset[ctrl_i]

        input_emb = torch.cat(
            [gene_emb_c, drug_emb_p, log1p_conc_p.unsqueeze(0)], dim=0
        )
        target_emb = gene_emb_p
        return input_emb, target_emb, drug_name_p


# --------------------- Training / eval utils ---------------------
def train_one_epoch(model, train_loader, criterion, optimizer, accelerator):
    model.train()
    total_train_loss = 0.0
    seen_train = 0

    for inputs, targets, drug_names in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        bs = inputs.size(0)
        total_train_loss += loss.item() * bs
        seen_train += bs

    avg_train_loss = total_train_loss / max(seen_train, 1)
    return avg_train_loss


def evaluate(model, val_loader, criterion):
    model.eval()
    total_val_loss = 0.0
    seen_val = 0

    with torch.no_grad():
        for inputs, targets, drug_names in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            bs = inputs.size(0)
            total_val_loss += loss.item() * bs
            seen_val += bs

    avg_val_loss = total_val_loss / max(seen_val, 1)
    return avg_val_loss


def preview_predictions(model, val_loader, accelerator):
    model.eval()
    with torch.no_grad():
        for inputs, targets, drug_names in val_loader:
            outputs = model(inputs)
            if accelerator.is_main_process:
                for i in range(min(5, inputs.size(0))):
                    print(f"Drug: {drug_names[i]}")
                    print(f"Target: {targets[i][:5]}")
                    print(f"Output: {outputs[i][:5]}")
            break


def save_full_predictions(dataset, model, accelerator, device, pred_out_path):
    if not accelerator.is_main_process:
        return

    full_loader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=4,
        shuffle=False,
        persistent_workers=True,
    )

    unwrapped_model = accelerator.unwrap_model(model).to(device).eval()
    predicted_gene_embs = []

    with torch.no_grad():
        for inputs, targets, drug_names in full_loader:
            inputs = inputs.to(device, non_blocking=True)
            z_pred = unwrapped_model(inputs)
            predicted_gene_embs.append(z_pred.detach().cpu().numpy())

    predicted_gene_embs = np.concatenate(predicted_gene_embs, axis=0)

    os.makedirs(os.path.dirname(pred_out_path), exist_ok=True)
    np.savez_compressed(pred_out_path, predicted_gene_embs=predicted_gene_embs)
    print("✅ Saved predicted embeddings npz:", pred_out_path)


def main():
    # --------------------- Config ---------------------
    split_path = "/home/mark/repos/genedrug/cache/dataset/tahoe100m/log1p/gaussian/paired_indices_split.np"
    drug_emb_csv = "/ml_storage/jiaxin/tahoe_drug_unimol_embeddings_v1.csv"
    gene_emb_path = "/ml_storage/jiaxin/catch_up/best_performance_ae/relmape/embedding/encoded/1772700225.0386295.pt"
    save_path = "/ml_storage/jiaxin/catch_up/best_performance_ae/relmape/pert_mlp_weight/perturbation_mlp_residual_swiglu_best_model.pt"
    pred_out_path = "/ml_storage/jiaxin/catch_up/best_performance_ae/benchmark_without_ae/embedding/predict_raw/residual_swiglu.npz"

    batch_size = 128
    num_workers = 4
    max_epochs = 300
    early_stop_patience = 15
    drug_ctrl_name = "DMSO_TF"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------- Build dataset meta (cell_line & drug vocab) ---------------------
    h5ad_paths = load_h5ad_paths()
    cell_line_to_id, drug_name_to_id = build_cellline_and_drug_vocab(h5ad_paths)
    print(f"Number of cell line labels: {len(cell_line_to_id)}")
    print(f"Number of drug name labels: {len(drug_name_to_id)}")

    # --------------------- Drug embeddings ---------------------
    drug_name_2_emb = load_drug_embeddings(drug_emb_csv)
    drug_embs_tensor = build_normalized_drug_emb_tensor(
        drug_name_to_id=drug_name_to_id,
        drug_name_2_emb=drug_name_2_emb,
        device=device,
    )
    print(f"Drug embeddings tensor shape: {drug_embs_tensor.shape}")

    # --------------------- Gene embeddings ---------------------
    gene_embs = torch.load(gene_emb_path, map_location="cpu")
    gene_dim = gene_embs.shape[1]

    # --------------------- Raw dataset ---------------------
    encode_raw_with_emb = make_encode_raw_with_emb(
        gene_embs=gene_embs,
        drug_name_to_id=drug_name_to_id,
        drug_name_2_emb=drug_name_2_emb,
    )

    dataset_raw = CachedGaussianDataset(transform=encode_raw_with_emb)
    _ = dataset_raw[0]

    # --------------------- Meta dataset / pairing ---------------------
    dataset_meta = CachedGaussianDataset(transform=encode_meta_only)
    drug_ctrl_idx = build_drug_ctrl_idx(dataset_meta, drug_ctrl_name=drug_ctrl_name)

    dataset = PairedGaussianDataset(
        raw_dataset=dataset_raw,
        meta_dataset=dataset_meta,
        drug_ctrl_idx=drug_ctrl_idx,
        drug_ctrl_name=drug_ctrl_name,
    )
    print("paired dataset len =", len(dataset))

    # --------------------- Train/Val split ---------------------
    train_dataset, val_dataset, test_dataset = get_split_datasets(
        dataset=dataset,
        split_path=split_path,
    )

    train_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        persistent_workers=True,
    )

    # --------------------- Train predictor ---------------------
    input_dim = gene_dim + 512 + 1
    output_dim = gene_dim

    model = ResidualGatedMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        model_dim=2048,
        n_blocks=6,
        hidden_mult=4,
        dropout=0.1,
        gated_act="swiglu",
    ).to(device)

    accelerator_obj = accelerate.Accelerator()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
        factor=0.5,
    )

    model, optimizer, train_loader, val_loader = accelerator_obj.prepare(
        model, optimizer, train_loader, val_loader
    )

    best_val_loss = float("inf")
    patience_counter = 0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(max_epochs):
        avg_train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            accelerator=accelerator_obj,
        )

        avg_val_loss = evaluate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
        )

        scheduler.step(avg_val_loss)

        if accelerator_obj.is_main_process:
            print(
                f"Epoch [{epoch + 1}/{max_epochs}], "
                f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            accelerator_obj.wait_for_everyone()
            unwrapped_model = accelerator_obj.unwrap_model(model)
            if accelerator_obj.is_main_process:
                torch.save(unwrapped_model.state_dict(), save_path)
                print("New model best weight saved")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                if accelerator_obj.is_main_process:
                    print("Early stopping triggered.")
                break

    if accelerator_obj.is_main_process:
        print("✅ Done. Saved model to:", save_path)

    # --------------------- Preview predictions (VAL only) ---------------------
    preview_predictions(
        model=model,
        val_loader=val_loader,
        accelerator=accelerator_obj,
    )

    # --------------------- Save predicted embeddings (z_pred) for FULL paired dataset ---------------------
    save_full_predictions(
        dataset=dataset,
        model=model,
        accelerator=accelerator_obj,
        device=device,
        pred_out_path=pred_out_path,
    )


if __name__ == "__main__":
    main()