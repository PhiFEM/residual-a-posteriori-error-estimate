import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import argparse

parent_dir = os.path.dirname(__file__)

def plot_h10(output_dir, cutoff=-4, style="-^"):
    ref_method = os.path.split(output_dir)[-1]
    solver = os.path.split(output_dir)[0].split("_")[-1]
    
    df = pd.read_csv(os.path.join(output_dir, "results.csv"))
    dofs = df["dofs"].values

    for key in [k for k in df.keys() if "H10" in k]:
        vals = df[key].values
        plt.loglog(dofs, vals, style, label=f"{key} ({solver}, {ref_method})")
        if cutoff < 0:
            a, b = np.polyfit(np.log(dofs[cutoff:]), np.log(vals[cutoff:]), 1)
            plt.loglog(dofs[cutoff:], 0.8*np.exp(b)*dofs[cutoff:]**a, "--", color='black', label=f"Slope {np.round(a,2)}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="plot",
                                    description="Generate a loglog convergence plot.")

    parser.add_argument("mode", type=str, choices=["FEM", "phiFEM", "comparison"])
    args = parser.parse_args()
    mode = args.mode

    plt.figure()
    if mode in ["FEM", "phiFEM"]:
        save_dir = os.path.join(parent_dir, "output" + "_" + mode)

        output_dir = os.path.join(parent_dir, "output" + "_" + mode, "adaptive")
        plot_h10(output_dir, style="-o", cutoff=-10)
        # output_dir = os.path.join(parent_dir, "output" + "_" + fe_method, "omega_h")
        # plot_h10(output_dir, style="--^")

    elif mode=="comparison":
        save_dir = os.path.join(parent_dir)

        output_dir = os.path.join(parent_dir, "output_FEM", "adaptive")
        plot_h10(output_dir, style="-o", cutoff=-10)
        output_dir = os.path.join(parent_dir, "output_phiFEM", "adaptive")
        plot_h10(output_dir, style="--^", cutoff=-10)
    
    plt.xlabel("dofs")
    plt.ylabel(r"$\eta$")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "plot_adaptive.pdf"), bbox_inches="tight")
