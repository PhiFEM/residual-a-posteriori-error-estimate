import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

parent_dir = os.path.dirname(__file__)

def plot_h10(ref_method="background", cutoff=-4, style="-^"):
    df = pd.read_csv(os.path.join(parent_dir, "output", ref_method, "results.csv"))
    dofs = df["dofs"].values

    for key in [k for k in df.keys() if "H10" in k]:
        vals = df[key].values
        plt.loglog(dofs, vals, style, label=f"{key} ({ref_method})")
        if cutoff < 0:
            a, b = np.polyfit(np.log(dofs[cutoff:]), np.log(vals[cutoff:]), 1)
            plt.loglog(dofs[cutoff:], 0.8*np.exp(b)*dofs[cutoff:]**a, "--", color='black', label=f"Slope {np.round(a,2)}")

if __name__=="__main__":
    plt.figure()

    plot_h10(ref_method="omega_h", style="-o")
    plot_h10(ref_method="adaptive", style="--^", cutoff=-10)

    plt.xlabel("dofs")
    plt.ylabel(r"$\eta$")
    plt.legend()
    plt.savefig(os.path.join(parent_dir, "output", "plot_adaptive.pdf"), bbox_inches="tight")