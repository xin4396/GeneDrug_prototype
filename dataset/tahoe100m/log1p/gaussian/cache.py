# %%
import os
from cfg import PATH_PREFIX

import numpy as np
import numpy.typing as npt


def get_log1p_mean_std(
    args,
) -> tuple[npt.NDArray[np.float32] | None, npt.NDArray[np.float32] | None]:
    plate, cell_line_code, drugname_drugconc_code = args
    h5ad_path = f"{PATH_PREFIX}/datasets/Tahoe100M_Original/plate{plate}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"

    from dataset.tahoe100m.h5ad import H5ADDataset

    dataset = H5ADDataset(
        h5ad_path=h5ad_path,
        component_keys=[
            ("obs", "cell_line"),
            ("obs", "drugname_drugconc"),
        ],
    )

    indices = np.where(
        (dataset.h5f["obs"]["cell_line"]["codes"][:] == cell_line_code)
        & (
            dataset.h5f["obs"]["drugname_drugconc"]["codes"][:]
            == drugname_drugconc_code
        )
    )[0]

    if len(indices) == 0:
        return None, None

    # compute mean and stddev of gene expression for these indices
    gene_expressions = np.array(
        [dataset.get_X_i(idx) for idx in indices]
    )  # shape: (num_samples, num_genes)
    log1p_gene_expressions = np.log1p(gene_expressions)
    log1p_mean = np.mean(log1p_gene_expressions, axis=0)
    log1p_std = np.std(log1p_gene_expressions, axis=0)

    return log1p_mean, log1p_std


# %%
import tqdm

args_list = []

for plate in tqdm.tqdm(range(1, 15)):
    from dataset.tahoe100m.h5ad import H5ADDataset

    dataset_i = H5ADDataset(
        h5ad_path=f"{PATH_PREFIX}/datasets/Tahoe100M_Original/plate{plate}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
        component_keys=[
            ("obs", "cell_line"),
            ("obs", "drugname_drugconc"),
        ],
    )

    cell_line_bytes_i = dataset_i.h5f["obs"]["cell_line"]["categories"][:]
    cell_line_i_cnt = 0
    for cell_line_byte in cell_line_bytes_i:
        cell_line_name = cell_line_byte.decode("utf-8")
        cell_line_code = cell_line_i_cnt
        cell_line_i_cnt += 1

        drugname_drugconc_bytes_i = dataset_i.h5f["obs"]["drugname_drugconc"][
            "categories"
        ][:]
        drugname_drugconc_i_cnt = 0
        for drugname_drugconc_byte in drugname_drugconc_bytes_i:
            drugname_drugconc_str = drugname_drugconc_byte.decode("utf-8")
            drugname_drugconc_code = drugname_drugconc_i_cnt
            drugname_drugconc_i_cnt += 1

            # drug_name, drug_conc = eval(drugname_drugconc_str)[0][:2]

            # mu, sigma = get_mu_sigma(i, cell_line_code, drugname_drugconc_code)

            args_list.append((plate, cell_line_code, drugname_drugconc_code))

# %%
import multiprocessing

with multiprocessing.Pool(processes=os.cpu_count() // 2) as pool:
    results: list[
        tuple[npt.NDArray[np.float32] | None, npt.NDArray[np.float32] | None]
    ] = list(tqdm.tqdm(pool.imap(get_log1p_mean_std, args_list), total=len(args_list)))

# %%
log1p_mean = []
log1p_std = []
for i in results:
    log1p_mean.append(i[0])
    log1p_std.append(i[1])

import numpy as np

mask = []
for i, m in enumerate(log1p_mean):
    if m is None:
        mask.append(False)
        log1p_mean[i] = np.zeros((62710,))
        log1p_std[i] = np.zeros((62710,))
    else:
        mask.append(True)

# %%
log1p_mean = np.array(log1p_mean)

# %%
log1p_std = np.array(log1p_std)

# %%
mask = np.array(mask)

# %%
res_path = "cache/dataset/tahoe100m/logfc/gaussian"
os.makedirs(res_path, exist_ok=True)

# %%
np.savez_compressed(
    f"{res_path}/result.npz",
    log1p_mean=log1p_mean,
    log1p_std=log1p_std,
    mask=mask,
)

# %%
cnt_none = 0
for r in results:
    if r[0] is None and r[1] is None:
        cnt_none += 1
print(f"Number of (mu, sigma) pairs not found: {cnt_none}")
# %%
# import pickle

# with open(f"{res_path}/args.pkl", "wb") as f:
#     pickle.dump(args_list, f)
# # %%
# import pickle
# import pgzip

# with open(f"{res_path}/gaussian_mp_args.pkl", "rb") as f:
#     args_list = pickle.load(f)

# # with open(f"{res_path}/gaussian_mp_results.pkl", "rb") as f:
# #     results = pickle.load(f)

# with pgzip.open("out/classifier_380/gaussian_mp_results.pkl.gz", "rb") as f:
#     results = pickle.load(f)
# %%
cell_line_label_to_str: dict[int, dict[int, str]] = {}
drugname_drugconc_label_to_str: dict[int, dict[int, str]] = {}

for plate in tqdm.tqdm(range(1, 15)):
    from dataset.tahoe100m.h5ad import H5ADDataset

    dataset_i = H5ADDataset(
        h5ad_path=f"{PATH_PREFIX}/datasets/Tahoe100M_Original/plate{plate}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
        component_keys=[
            ("obs", "cell_line"),
            ("obs", "drugname_drugconc"),
        ],
    )

    cell_line_bytes_i = dataset_i.h5f["obs"]["cell_line"]["categories"][:]
    cell_line_i_cnt = 0
    cell_line_i_label_to_str = {}
    for cell_line_byte in cell_line_bytes_i:
        cell_line_name = cell_line_byte.decode("utf-8")
        cell_line_code = cell_line_i_cnt
        cell_line_i_label_to_str[cell_line_code] = cell_line_name
        cell_line_i_cnt += 1

    drugname_drugconc_i_label_to_str = {}
    drugname_drugconc_bytes_i = dataset_i.h5f["obs"]["drugname_drugconc"]["categories"][
        :
    ]
    drugname_drugconc_i_cnt = 0
    for drugname_drugconc_byte in drugname_drugconc_bytes_i:
        drugname_drugconc_str = drugname_drugconc_byte.decode("utf-8")
        drugname_drugconc_code = drugname_drugconc_i_cnt
        drugname_drugconc_i_label_to_str[drugname_drugconc_code] = drugname_drugconc_str
        drugname_drugconc_i_cnt += 1

        # drug_name, drug_conc = eval(drugname_drugconc_str)[0][:2]

    cell_line_label_to_str[plate] = cell_line_i_label_to_str
    drugname_drugconc_label_to_str[plate] = drugname_drugconc_i_label_to_str

# %%
args_str_list = []
for idx, args in enumerate(args_list):
    plate, cell_line_code, drugname_drugconc_code = args
    cell_line_str = cell_line_label_to_str[plate][cell_line_code]
    drugname_drugconc_str = drugname_drugconc_label_to_str[plate][
        drugname_drugconc_code
    ]

    drug_name, drug_conc = eval(drugname_drugconc_str)[0][:2]
    args_str_list.append((plate, cell_line_str, drug_name, drug_conc))
# %%
import pickle

with open(f"{res_path}/args_str.pkl", "wb") as f:
    pickle.dump(args_str_list, f)

# %%
