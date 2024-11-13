import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def plot_h10(output_dir, ref_method, plot_kwargs, cutoff=-4):
    df = pd.read_csv(os.path.join(output_dir, ref_method, "results.csv"))
    dofs    = df["dofs"].values
    h10_est = df["H10 estimator"].values

    plt.loglog(dofs, h10_est, "--", **plot_kwargs)
    if cutoff < 0:
        e, f = np.polyfit(np.log(dofs[cutoff:]), np.log(h10_est[cutoff:]), 1)
        plt.loglog(dofs[cutoff:], 0.8*np.exp(f)*dofs[cutoff:]**e, "--", color='black', label=f"Slope {np.round(e,2)}")

if __name__=="__main__":
    plt.figure()

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    plot_kwargs = {"color": "c",
                   "alpha": 1,
                   "marker": "o",
                   "label": "H10 estimator (uniform, FEM)"}
    plot_h10(output_dir, "omega_h", plot_kwargs, cutoff=1)
    plot_kwargs = {"color": "c",
                   "alpha": 1,
                   "marker": "^",
                   "label": "H10 estimator (adaptive, FEM)"}
    plot_h10(output_dir, "adaptive", plot_kwargs, cutoff=1)

    output_dir = os.path.join("../", "lshaped", "output", "submesh")
    plot_kwargs = {"color": "blue",
                   "marker": "o",
                   "label": "H10 estimator (uniform, phiFEM)"}
    plot_h10(output_dir, "omega_h", plot_kwargs, cutoff=1)
    plot_kwargs = {"color": "blue",
                   "marker": "^",
                   "label": "H10 estimator (adaptive, phiFEM)"}
    plot_h10(output_dir, "adaptive", plot_kwargs, cutoff=1)

    plt.xlabel("dofs")
    plt.ylabel(r"$\eta$")
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), "output", "plot_adaptive.pdf"), bbox_inches="tight")
    plt.figure()