import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

parent_dir = os.path.dirname(__file__)

def plot_h10(mesh_type="bg_mesh", ref_method="background", cutoff=-4, marker="^"):
    df = pd.read_csv(os.path.join(parent_dir, "output", mesh_type, ref_method, "results.csv"))
    dofs    = df["dofs"].values
    h10_est = df["H10 estimator"].values

    plt.loglog(dofs, h10_est, "--" + marker, label=f"H10 estimator ({mesh_type}, {ref_method})")
    if cutoff < 0:
        e, f = np.polyfit(np.log(dofs[cutoff:]), np.log(h10_est[cutoff:]), 1)
        plt.loglog(dofs[cutoff:], 0.8*np.exp(f)*dofs[cutoff:]**e, "--", color='black', label=f"Slope {np.round(e,2)}")

if __name__=="__main__":
    plt.figure()

    plot_h10(mesh_type="bg_mesh", ref_method="omega_h", marker="o")
    plot_h10(mesh_type="bg_mesh", ref_method="adaptive", marker="^", cutoff=-10)

    plt.xlabel("dofs")
    plt.ylabel(r"$\eta$")
    plt.legend()
    plt.savefig(os.path.join(parent_dir, "output", "bg_mesh", "plot_adaptive.pdf"), bbox_inches="tight")
    plt.figure()

    plot_h10(mesh_type="submesh", ref_method="omega_h", marker="o")
    plot_h10(mesh_type="submesh", ref_method="adaptive", marker="^", cutoff=-10)

    plt.xlabel("dofs")
    plt.ylabel(r"$\eta$")
    plt.legend()
    plt.savefig(os.path.join(parent_dir, "output", "submesh", "plot_adaptive.pdf"), bbox_inches="tight")