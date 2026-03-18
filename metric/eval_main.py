# %%
import torch
import numpy as np
from scipy import stats
import warnings
from typing import Dict, Tuple, Optional


def compute_perturbation_metrics(
    dataloader, top_k_list: list = [200, 1000], device: str = "cpu"
) -> Dict[str, float]:
    """
    Compute comprehensive perturbation prediction metrics.

    Args:
        dataloader: PyTorch DataLoader providing (x_pre, x_post_true, x_post_pred)
        top_k_list: List of k values for DEG accuracy metrics
        device: Device to use for computations

    Returns:
        Dictionary containing all computed metrics
    """

    # Collect all data
    x_pre_list, x_post_true_list, x_post_pred_list = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            x_pre, x_post_true, x_post_pred = batch
            x_pre_list.append(x_pre)
            x_post_true_list.append(x_post_true)
            x_post_pred_list.append(x_post_pred)

    # Concatenate all batches
    x_pre = torch.cat(x_pre_list, dim=0).to(device)
    x_post_true = torch.cat(x_post_true_list, dim=0).to(device)
    x_post_pred = torch.cat(x_post_pred_list, dim=0).to(device)

    # Verify shapes
    assert x_pre.shape == x_post_true.shape == x_post_pred.shape
    n_samples, n_genes = x_pre.shape

    metrics = {}

    # 1. Convert from log1p space to count space
    # Note: x = mean(log1p(count)) over cells
    # To get back to approximate counts: exp(x) - 1
    x_pre_counts = torch.exp(x_pre) - 1
    x_post_true_counts = torch.exp(x_post_true) - 1
    x_post_pred_counts = torch.exp(x_post_pred) - 1

    # 2. Compute log fold-changes
    logfc_true = x_post_true - x_pre  # in log1p space
    logfc_pred = x_post_pred - x_pre

    # 3. Compute perturbation shifts (Δ)
    delta_true = x_post_true_counts - x_pre_counts  # in count space
    delta_pred = x_post_pred_counts - x_pre_counts

    # Function to compute correlations safely
    def safe_correlation(x, y, method="pearson"):
        """Compute correlation with NaN handling."""
        if x.shape[0] <= 1:
            return float("nan")

        # Convert to numpy for scipy stats
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()

        # Remove NaN/inf values
        mask = np.isfinite(x_np) & np.isfinite(y_np)
        if np.sum(mask) < 2:
            return float("nan")

        x_clean = x_np[mask]
        y_clean = y_np[mask]

        if method == "pearson":
            return stats.pearsonr(x_clean, y_clean)[0]
        else:  # spearman
            return stats.spearmanr(x_clean, y_clean)[0]

    # 4. Compute logFC correlations (full gene set)
    logfc_pearson = safe_correlation(
        logfc_true.flatten(), logfc_pred.flatten(), "pearson"
    )
    logfc_spearman = safe_correlation(
        logfc_true.flatten(), logfc_pred.flatten(), "spearman"
    )

    metrics["logFC-Pearson"] = logfc_pearson
    metrics["logFC-Spearman"] = logfc_spearman

    # 5. Compute Δ correlations (full gene set)
    delta_pearson = safe_correlation(
        delta_true.flatten(), delta_pred.flatten(), "pearson"
    )
    delta_spearman = safe_correlation(
        delta_true.flatten(), delta_pred.flatten(), "spearman"
    )

    metrics["Δ-Pearson"] = delta_pearson
    metrics["Δ-Spearman"] = delta_spearman

    # 6. Identify DEGs (Differentially Expressed Genes)
    # Using absolute log fold-change to identify top DEGs
    abs_logfc = torch.abs(logfc_true)

    # For per-sample DEG identification
    deg_indices = []
    for i in range(n_samples):
        # Get top genes by absolute logFC
        _, top_indices = torch.topk(abs_logfc[i], k=min(2000, n_genes))
        deg_indices.append(top_indices)

    # 7. Compute DEG-based metrics
    deg_logfc_pearsons, deg_logfc_spearmans = [], []
    deg_delta_pearsons, deg_delta_spearmans = [], []

    for i in range(n_samples):
        deg_idx = deg_indices[i]

        if len(deg_idx) >= 2:  # Need at least 2 points for correlation
            # logFC correlations for DEGs
            corr_pearson = safe_correlation(
                logfc_true[i, deg_idx], logfc_pred[i, deg_idx], "pearson"
            )
            corr_spearman = safe_correlation(
                logfc_true[i, deg_idx], logfc_pred[i, deg_idx], "spearman"
            )
            if not np.isnan(corr_pearson):
                deg_logfc_pearsons.append(corr_pearson)
            if not np.isnan(corr_spearman):
                deg_logfc_spearmans.append(corr_spearman)

            # Δ correlations for DEGs
            corr_pearson = safe_correlation(
                delta_true[i, deg_idx], delta_pred[i, deg_idx], "pearson"
            )
            corr_spearman = safe_correlation(
                delta_true[i, deg_idx], delta_pred[i, deg_idx], "spearman"
            )
            if not np.isnan(corr_pearson):
                deg_delta_pearsons.append(corr_pearson)
            if not np.isnan(corr_spearman):
                deg_delta_spearmans.append(corr_spearman)

    # Average across samples
    metrics["logFC-Pearson(DEG)"] = (
        np.nanmean(deg_logfc_pearsons) if deg_logfc_pearsons else float("nan")
    )
    metrics["logFC-Spearman(DEG)"] = (
        np.nanmean(deg_logfc_spearmans) if deg_logfc_spearmans else float("nan")
    )
    metrics["Δ-Pearson(DEG)"] = (
        np.nanmean(deg_delta_pearsons) if deg_delta_pearsons else float("nan")
    )
    metrics["Δ-Spearman(DEG)"] = (
        np.nanmean(deg_delta_spearmans) if deg_delta_spearmans else float("nan")
    )

    # 8. Compute Explained Variance (EV)
    # EV = 1 - var(delta) / variance
    def explained_variance(y_true, y_pred, axis=0):
        """Compute explained variance along specified axis."""
        variance = torch.var(y_true, dim=axis)
        delta_variance = torch.var((y_true - y_pred), dim=axis)

        # Avoid division by zero
        mask = variance > 1e-10
        ev = torch.zeros_like(variance)
        ev[mask] = 1 - (delta_variance[mask] / variance[mask])

        # Clip to valid range [-inf, 1]
        ev = torch.clamp(ev, min=-10, max=1)
        return ev

    # EV per gene (across samples)
    ev_per_gene = explained_variance(x_post_true, x_post_pred, axis=0)
    metrics["EV_median"] = (
        torch.median(ev_per_gene[ev_per_gene.isfinite()]).item()
        if torch.any(ev_per_gene.isfinite())
        else float("nan")
    )

    # EV for DEGs
    # Use top DEGs across all samples (average absolute logFC)
    avg_abs_logfc = torch.mean(abs_logfc, dim=0)
    top_k = min(1000, n_genes)
    _, top_deg_indices = torch.topk(avg_abs_logfc, k=top_k)

    if len(top_deg_indices) > 0:
        ev_deg = explained_variance(
            x_post_true[:, top_deg_indices], x_post_pred[:, top_deg_indices], axis=0
        )
        metrics["EV_median(DEG)"] = (
            torch.median(ev_deg[ev_deg.isfinite()]).item()
            if torch.any(ev_deg.isfinite())
            else float("nan")
        )
    else:
        metrics["EV_median(DEG)"] = float("nan")

    # 9. Compute DEG-accuracy (Overlap accuracy)
    for top_k in top_k_list:
        if top_k > n_genes:
            metrics[f"DEG-accuracy(top{top_k})"] = float("nan")
            continue

        accuracies = []
        for i in range(n_samples):
            # Get top-k DEGs by absolute logFC
            _, true_topk = torch.topk(abs_logfc[i], k=top_k)
            _, pred_topk = torch.topk(torch.abs(logfc_pred[i]), k=top_k)

            # Convert to sets and compute overlap
            true_set = set(true_topk.cpu().numpy())
            pred_set = set(pred_topk.cpu().numpy())

            overlap = len(true_set.intersection(pred_set))
            accuracy = overlap / top_k
            accuracies.append(accuracy)

        metrics[f"DEG-accuracy(top{top_k})"] = (
            np.mean(accuracies) if accuracies else float("nan")
        )

    # 10. Report any metrics that couldn't be calculated
    nan_metrics = [k for k, v in metrics.items() if np.isnan(v)]
    if nan_metrics:
        warnings.warn(f"The following metrics could not be calculated: {nan_metrics}")

    return metrics


def print_metrics_summary(metrics: Dict[str, float]):
    """Print formatted metrics summary."""
    print("=" * 60)
    print("Perturbation Prediction Metrics Summary")
    print("=" * 60)

    categories = {
        "logFC Correlations": [
            "logFC-Pearson",
            "logFC-Spearman",
            "logFC-Pearson(DEG)",
            "logFC-Spearman(DEG)",
        ],
        "Δ Correlations": [
            "Δ-Pearson",
            "Δ-Spearman",
            "Δ-Pearson(DEG)",
            "Δ-Spearman(DEG)",
        ],
        "Explained Variance": ["EV_median", "EV_median(DEG)"],
        "DEG Accuracy": [k for k in metrics.keys() if "DEG-accuracy" in k],
    }

    for category, metric_list in categories.items():
        print(f"\n{category}:")
        print("-" * 40)
        for metric_name in metric_list:
            if metric_name in metrics:
                value = metrics[metric_name]
                if np.isnan(value):
                    print(f"  {metric_name:30s}: Could not be calculated")
                else:
                    print(f"  {metric_name:30s}: {value:.4f}")

    print("=" * 60)


from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def compute_perturbation_metrics_batched(
    dataloader,
    top_k_list: List[int] = [200, 1000],
    deg_k: int = 200,  # For DEG-based correlations
    device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Compute perturbation prediction metrics in a batched/streaming fashion.

    Args:
        dataloader: PyTorch DataLoader providing (x_pre, x_post_true, x_post_pred)
        top_k_list: List of k values for DEG accuracy metrics
        deg_k: Number of top DEGs to use for DEG-based metrics
        device: Device to use for computations
        verbose: Print progress information

    Returns:
        Dictionary containing all computed metrics
    """

    # Initialize accumulators for streaming computation
    metrics = defaultdict(list)
    ev_accumulators = {}
    deg_global_accumulators = {}

    # For global DEG identification
    global_abs_logfc_sum = None
    global_sample_count = 0

    # Process batches
    batch_idx = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if verbose:  # and batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}...")

            x_pre, x_post_true, x_post_pred = batch
            batch_size = x_pre.shape[0]
            n_genes = x_pre.shape[1]
            total_samples += batch_size

            # Move to device
            x_pre = x_pre.to(device)
            x_post_true = x_post_true.to(device)
            x_post_pred = x_post_pred.to(device)

            # 1. Convert from log1p space to count space
            x_pre_counts = torch.exp(x_pre) - 1
            x_post_true_counts = torch.exp(x_post_true) - 1
            x_post_pred_counts = torch.exp(x_post_pred) - 1

            # 2. Compute log fold-changes and delta vectors
            logfc_true = x_post_true - x_pre  # in log1p space
            logfc_pred = x_post_pred - x_pre
            delta_true = x_post_true_counts - x_pre_counts  # in count space
            delta_pred = x_post_pred_counts - x_pre_counts

            # Update global DEG statistics (average absolute logFC)
            batch_abs_logfc = torch.abs(logfc_true).mean(dim=0)
            if global_abs_logfc_sum is None:
                global_abs_logfc_sum = batch_abs_logfc
            else:
                # Weighted average update
                global_abs_logfc_sum = (
                    global_abs_logfc_sum * global_sample_count
                    + batch_abs_logfc * batch_size
                ) / (global_sample_count + batch_size)
            global_sample_count += batch_size

            # 3. Compute correlations per sample in batch
            for i in range(batch_size):
                # Full gene set correlations
                logfc_pearson = safe_correlation(
                    logfc_true[i], logfc_pred[i], "pearson"
                )
                logfc_spearman = safe_correlation(
                    logfc_true[i], logfc_pred[i], "spearman"
                )

                delta_pearson = safe_correlation(
                    delta_true[i], delta_pred[i], "pearson"
                )
                delta_spearman = safe_correlation(
                    delta_true[i], delta_pred[i], "spearman"
                )

                if not np.isnan(logfc_pearson):
                    metrics["logFC-Pearson"].append(logfc_pearson)
                if not np.isnan(logfc_spearman):
                    metrics["logFC-Spearman"].append(logfc_spearman)
                if not np.isnan(delta_pearson):
                    metrics["Δ-Pearson"].append(delta_pearson)
                if not np.isnan(delta_spearman):
                    metrics["Δ-Spearman"].append(delta_spearman)

                # DEG-based correlations
                abs_logfc_i = torch.abs(logfc_true[i])
                top_deg_indices = torch.topk(abs_logfc_i, k=min(deg_k, n_genes)).indices

                if len(top_deg_indices) >= 2:
                    deg_logfc_pearson = safe_correlation(
                        logfc_true[i][top_deg_indices],
                        logfc_pred[i][top_deg_indices],
                        "pearson",
                    )
                    deg_logfc_spearman = safe_correlation(
                        logfc_true[i][top_deg_indices],
                        logfc_pred[i][top_deg_indices],
                        "spearman",
                    )
                    deg_delta_pearson = safe_correlation(
                        delta_true[i][top_deg_indices],
                        delta_pred[i][top_deg_indices],
                        "pearson",
                    )
                    deg_delta_spearman = safe_correlation(
                        delta_true[i][top_deg_indices],
                        delta_pred[i][top_deg_indices],
                        "spearman",
                    )

                    if not np.isnan(deg_logfc_pearson):
                        metrics["logFC-Pearson(DEG)"].append(deg_logfc_pearson)
                    if not np.isnan(deg_logfc_spearman):
                        metrics["logFC-Spearman(DEG)"].append(deg_logfc_spearman)
                    if not np.isnan(deg_delta_pearson):
                        metrics["Δ-Pearson(DEG)"].append(deg_delta_pearson)
                    if not np.isnan(deg_delta_spearman):
                        metrics["Δ-Spearman(DEG)"].append(deg_delta_spearman)

                # DEG accuracy metrics
                for k in top_k_list:
                    if k > n_genes:
                        continue

                    # Get top-k DEGs for true and predicted
                    true_topk = torch.topk(abs_logfc_i, k=k).indices
                    pred_topk = torch.topk(torch.abs(logfc_pred[i]), k=k).indices

                    # Compute overlap
                    true_set = set(true_topk.cpu().numpy())
                    pred_set = set(pred_topk.cpu().numpy())
                    overlap = len(true_set.intersection(pred_set))
                    accuracy = overlap / k

                    metrics[f"DEG-accuracy(top{k})"].append(accuracy)

            # 4. Accumulate statistics for Explained Variance
            # EV per gene requires global statistics
            if len(ev_accumulators) == 0:
                ev_accumulators = {
                    "sum_true": torch.zeros(n_genes, device=device),
                    "sum_true_sq": torch.zeros(n_genes, device=device),
                    "sum_squared_error": torch.zeros(n_genes, device=device),
                    "count": 0,
                }

            # Update accumulators
            ev_accumulators["sum_true"] += x_post_true.sum(dim=0)
            ev_accumulators["sum_true_sq"] += (x_post_true**2).sum(dim=0)
            ev_accumulators["sum_squared_error"] += (
                (x_post_true - x_post_pred) ** 2
            ).sum(dim=0)
            ev_accumulators["count"] += batch_size

            batch_idx += 1

    # 5. Compute final metrics from accumulators
    final_metrics = {}

    # Average correlation metrics
    correlation_metrics = [
        "logFC-Pearson",
        "logFC-Spearman",
        "Δ-Pearson",
        "Δ-Spearman",
        "logFC-Pearson(DEG)",
        "logFC-Spearman(DEG)",
        "Δ-Pearson(DEG)",
        "Δ-Spearman(DEG)",
    ]

    for metric_name in correlation_metrics:
        if metric_name in metrics and len(metrics[metric_name]) > 0:
            final_metrics[metric_name] = np.nanmean(metrics[metric_name])
        else:
            final_metrics[metric_name] = float("nan")

    # Average DEG accuracy metrics
    for k in top_k_list:
        metric_name = f"DEG-accuracy(top{k})"
        if metric_name in metrics and len(metrics[metric_name]) > 0:
            final_metrics[metric_name] = np.mean(metrics[metric_name])
        else:
            final_metrics[metric_name] = float("nan")

    # 6. Compute Explained Variance metrics
    if ev_accumulators["count"] > 0:
        n_genes = ev_accumulators["sum_true"].shape[0]

        # Compute variance and MSE per gene
        mean_true = ev_accumulators["sum_true"] / ev_accumulators["count"]
        variance = (ev_accumulators["sum_true_sq"] / ev_accumulators["count"]) - (
            mean_true**2
        )
        mse = ev_accumulators["sum_squared_error"] / ev_accumulators["count"]

        # Compute explained variance per gene
        ev_per_gene = torch.zeros(n_genes, device=device)
        mask = variance > 1e-10
        ev_per_gene[mask] = 1 - (mse[mask] / variance[mask])
        ev_per_gene[~mask] = torch.nan

        # Median EV (all genes)
        valid_ev = ev_per_gene[ev_per_gene.isfinite()]
        if len(valid_ev) > 0:
            final_metrics["EV_median"] = torch.median(valid_ev).item()
        else:
            final_metrics["EV_median"] = float("nan")

        # Median EV for DEGs
        if global_abs_logfc_sum is not None:
            # Get top-k DEGs based on global average absolute logFC
            top_k_global = min(deg_k, n_genes)
            _, top_deg_indices = torch.topk(global_abs_logfc_sum, k=top_k_global)

            ev_deg = ev_per_gene[top_deg_indices]
            valid_ev_deg = ev_deg[ev_deg.isfinite()]

            if len(valid_ev_deg) > 0:
                final_metrics["EV_median(DEG)"] = torch.median(valid_ev_deg).item()
            else:
                final_metrics["EV_median(DEG)"] = float("nan")
        else:
            final_metrics["EV_median(DEG)"] = float("nan")
    else:
        final_metrics["EV_median"] = float("nan")
        final_metrics["EV_median(DEG)"] = float("nan")

    # 7. Report any metrics that couldn't be calculated
    nan_metrics = [k for k, v in final_metrics.items() if np.isnan(v)]
    if nan_metrics and verbose:
        warnings.warn(f"The following metrics could not be calculated: {nan_metrics}")

    if verbose:
        print(f"Processed {total_samples} samples in {batch_idx} batches")

    return final_metrics


def safe_correlation(
    x: torch.Tensor, y: torch.Tensor, method: str = "pearson"
) -> float:
    """
    Compute correlation with NaN handling.

    Args:
        x: First tensor
        y: Second tensor
        method: 'pearson' or 'spearman'

    Returns:
        Correlation coefficient or NaN if can't compute
    """
    if x.shape[0] <= 1:
        return float("nan")

    # Convert to numpy for scipy stats
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    # Remove NaN/inf values
    mask = np.isfinite(x_np) & np.isfinite(y_np)
    if np.sum(mask) < 2:
        return float("nan")

    x_clean = x_np[mask]
    y_clean = y_np[mask]

    # Check if all values are constant
    if np.std(x_clean) < 1e-10 or np.std(y_clean) < 1e-10:
        return float("nan")

    try:
        if method == "pearson":
            return stats.pearsonr(x_clean, y_clean)[0]
        else:  # spearman
            return stats.spearmanr(x_clean, y_clean)[0]
    except:
        return float("nan")


class StreamingPearsonCorrelation:
    """Compute Pearson correlation in a streaming fashion."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_xy = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0

    def update(self, x_batch, y_batch):
        """Update with new batch of data."""
        batch_size = len(x_batch)
        self.n += batch_size
        self.sum_x += np.sum(x_batch)
        self.sum_y += np.sum(y_batch)
        self.sum_xy += np.sum(x_batch * y_batch)
        self.sum_x2 += np.sum(x_batch**2)
        self.sum_y2 += np.sum(y_batch**2)

    def compute(self):
        """Compute current correlation coefficient."""
        if self.n < 2:
            return float("nan")

        numerator = self.sum_xy - (self.sum_x * self.sum_y) / self.n
        denominator = np.sqrt(
            (self.sum_x2 - (self.sum_x**2) / self.n)
            * (self.sum_y2 - (self.sum_y**2) / self.n)
        )

        if abs(denominator) < 1e-10:
            return float("nan")

        return numerator / denominator


class StreamingSpearmanCorrelation:
    """Compute Spearman correlation in a streaming fashion using approximate rank statistics."""

    def __init__(self, n_bins=1000):
        self.n_bins = n_bins
        self.reset()

    def reset(self):
        self.data = []

    def update(self, x_batch, y_batch):
        """Update with new batch of data."""
        # For large datasets, we need to store all data for accurate ranks
        # For true streaming, consider using approximate quantile algorithms
        self.data.append((x_batch, y_batch))

    def compute(self):
        """Compute Spearman correlation."""
        if not self.data:
            return float("nan")

        # Concatenate all batches
        all_x = np.concatenate([d[0] for d in self.data])
        all_y = np.concatenate([d[1] for d in self.data])

        if len(all_x) < 2:
            return float("nan")

        # Compute ranks
        try:
            return stats.spearmanr(all_x, all_y)[0]
        except:
            return float("nan")

# %%
if __name__ == "__main__":
    h5ad_paths = [
        f"/ml_storage/ZMK/datasets/Tahoe100M_Original/plate{i}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
        for i in range(1, 15)
    ]

    from dataset.tahoe100m.h5ad import MultiH5ADDataset

    dataset = MultiH5ADDataset(
        h5ad_paths=h5ad_paths,
        component_keys=[
            ("obs", "cell_line"),
            ("obs", "drugname_drugconc"),
        ],
    )

    # %%
    cell_lines = set()
    for i in range(len(dataset.datasets)):
        cell_lines.update(
            dataset.datasets[i].h5f["obs"]["cell_line"]["categories"][:].astype(str)
        )

    cell_line_to_id = {}
    for i, cl in enumerate(sorted(cell_lines)):
        cell_line_to_id[cl] = i
    print(f"Number of cell line labels: {len(cell_line_to_id)}")

    # %%
    drug_names: set[str] = set()
    drugname_drugconc_bytes: set[bytes] = set()
    for i in range(len(dataset.datasets)):
        drugname_drugconc_bytes.update(
            dataset.datasets[i].h5f["obs"]["drugname_drugconc"]["categories"][:]
        )
    for s in drugname_drugconc_bytes:
        drug_name, drug_conc = eval(s.decode("utf-8"))[0][:2]
        drug_name = drug_name.strip()
        drug_names.add(drug_name)
    drug_name_to_id = {}
    for i, dn in enumerate(sorted(drug_names)):
        drug_name_to_id[dn] = i
    print(f"Number of drug name labels: {len(drug_name_to_id)}")

    # %%
    import pandas as pd
    import json
    import torch

    drug_emb_df = pd.read_csv(
        "/home/clab/Downloads/jiaxin_temporal/tahoe_drug_unimol_embeddings_v1.csv"
    )
    drug_name_2_emb = {}
    for _, row in drug_emb_df.iterrows():
        drug_name = row["drug"].strip()
        emb = torch.tensor(json.loads(row["embedding"]), dtype=torch.float32)
        drug_name_2_emb[drug_name] = emb

    drug_embs = []
    for drug_name in sorted(drug_name_to_id.keys()):
        drug_embs.append(drug_name_2_emb[drug_name])
    drug_embs_tensor = torch.stack(drug_embs)
    drug_embs_tensor = torch.nn.functional.normalize(drug_embs_tensor, p=2, dim=1)
    drug_embs_tensor = drug_embs_tensor.cuda()
    print(f"Drug embeddings tensor shape: {drug_embs_tensor.shape}")

    # %%
    from dataset.tahoe100m.log1p.gaussian.cached import CachedGaussianDataset

    gene_embs = torch.load(
        "/home/mark/repos/genedrug/cache/dataset/tahoe100m/log1p/gaussian/embedding/encoded/1767971613.6208487.pt"
    )

    def encode_data_point(data, index):
        gene_emb = gene_embs[index]
        drug_name = data["drug_name"].strip()
        drug_conc = torch.tensor(data["drug_conc"], dtype=torch.float32)
        cell_line_str = data["cell_line"]
        plate = data["plate"]
        gene_log1p_mean = data["log1p_mean"]
        return (
            plate,
            cell_line_str,
            drug_name,
            drug_conc,
            gene_emb,
            gene_log1p_mean,
        )

    dataset_raw = CachedGaussianDataset(transform=encode_data_point)

    # # %% to test autoencoder performance
    # with np.load(
    #     "cache/dataset/tahoe100m/log1p/gaussian/decoded/1767971613.6208487_mean.npz"
    # ) as data:
    #     gene_log1p_mean_decodes = data["decoded_means"]

    # %% to test drug purturbation prediction performance
    with np.load(
        "/home/clab/Downloads/jiaxin_temporal/catch_up/best_performance_ae/out/embedding/predicted/1767971613.6208487_residual_swiglu.npz"
    ) as data:
        gene_log1p_mean_preds = data["decoded_means"]

    # %%
    ## now process the dataset to filter out gene_ctrl to pair with gene_perturbed
    drug_ctrl_name = "DMSO_TF"
    ## save index of drug_ctrl for each plate, cell_line, drug_conc
    drug_ctrl_idx = {}
    for i in range(len(dataset_raw)):
        (
            plate,
            cell_line_str,
            drug_name,
            drug_conc,
            gene_emb,
            gene_log1p_mean,
        ) = dataset_raw[i]
        if drug_name == drug_ctrl_name:
            if plate not in drug_ctrl_idx:
                drug_ctrl_idx[plate] = {}
            drug_ctrl_idx[plate][cell_line_str] = i

    # %%
    ## now create new dataset with paired data points
    class PairedGaussianDataset(torch.utils.data.Dataset):
        def __init__(self, raw_dataset, drug_ctrl_idx):
            self.raw_dataset = raw_dataset
            self.drug_ctrl_idx = drug_ctrl_idx
            self.paired_indices = []
            for i in range(len(raw_dataset)):
                (
                    plate,
                    cell_line_str,
                    drug_name,
                    drug_conc,
                    gene_emb,
                    gene_log1p_mean,
                ) = raw_dataset[i]
                if drug_name != drug_ctrl_name:
                    if plate in drug_ctrl_idx and cell_line_str in drug_ctrl_idx[plate]:
                        ctrl_i = drug_ctrl_idx[plate][cell_line_str]
                        self.paired_indices.append((i, ctrl_i))

        def __len__(self):
            return len(self.paired_indices)

        def __getitem__(self, idx):
            perturbed_i, ctrl_i = self.paired_indices[idx]
            (
                plate_p,
                cell_line_str_p,
                drug_name_p,
                drug_conc_p,
                gene_emb_p,
                gene_log1p_mean_p,
            ) = self.raw_dataset[perturbed_i]
            (
                plate_c,
                cell_line_str_c,
                drug_name_c,
                drug_conc_c,
                gene_emb_c,
                gene_log1p_mean_c,
            ) = self.raw_dataset[ctrl_i]

            return (
                gene_log1p_mean_c,
                gene_log1p_mean_p,
                # gene_log1p_mean_decodes[perturbed_i],
                gene_log1p_mean_preds[idx],
            )

    dataset = PairedGaussianDataset(dataset_raw, drug_ctrl_idx)

    # %%
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=False)

    # %%
    # Compute metrics
    metrics = compute_perturbation_metrics_batched(
        dataloader, top_k_list=[200, 1000], device="cuda", verbose=True
    )

    # %%
    # Print summary
    print_metrics_summary(metrics)

# %%
