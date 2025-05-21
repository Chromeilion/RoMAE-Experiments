import torch
import matplotlib.pyplot as plt

from elasticc2.dataset import Elasticc2Dataset
from elasticc2.config import ElasticcConfig


def plot():
    plt.style.use('seaborn-v0_8-paper')
    plt.rc('text', usetex=True)
    plt.rc('text.latex')
    plt.rcParams["font.family"] = "Times New Roman"
    config = ElasticcConfig()
    with Elasticc2Dataset(
            config.dataset_location, split_no=0,
            split_type="training",
            gaussian_noise=config.gaussian_noise) as ds:
        sample = ds[0]
        bands = []
        for i in range(6):
            mask = torch.logical_and(sample["positions"][0] == i, ~sample["pad_mask"])
            band_times = sample["positions"][1][mask]
            band_values = sample["values"][mask, 0].squeeze()
            band_stds = sample["values"][mask, 1].squeeze()
            bands.append({"times": band_times, "values": band_values,
                          "stds": band_stds})
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
        fig, ax = plt.subplots(6, 1, figsize=(10, 8), sharex=True)
        handles, labels = [], []
        for i, band in enumerate(bands):
            ax[i].errorbar(band["times"], band["values"], yerr=band["stds"], fmt="o", color=colors[i], label=f"Band {i+1}")
            h, l = ax[i].get_legend_handles_labels()
            labels.extend(l)
            handles.extend(h)
        fig.tight_layout(rect=[0.04, 0.02, .88, 1])
        fig.legend(handles, labels, loc='outside right center', ncol=1)
        ax[-1].set_xlabel("Time (days)")
        fig.supylabel("Normalized Flux Difference")
        fig.subplots_adjust(hspace=0)
        fig.savefig("elasticc_fig.pdf")
