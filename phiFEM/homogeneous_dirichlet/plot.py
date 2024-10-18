import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(name, df):
    dofs = df["dofs"].values
    h10 = df["H10 error"].values
    l2 = df["L2 error"].values

    a, b = np.polyfit(np.log(dofs[:]), np.log(h10[:]), 1)
    c, d = np.polyfit(np.log(dofs[:]), np.log(l2[:]), 1)

    plt.loglog(dofs, h10, "-^", label=f"H10 error ({name})")
    plt.loglog(dofs, l2, "-^", label=f"L2 error ({name})")
    plt.loglog(dofs, 0.9*np.exp(b)*dofs[:]**a, "--", color='black', label=f"Slope {np.round(a,2)} ({name})")
    plt.loglog(dofs[:], 0.9*np.exp(d)*dofs[:]**c, "--", color='black', label=f"Slope {np.round(c,2)} ({name})")

plt.figure()

name = "fV"
df = pd.read_csv("results_fV.csv")
plot(name, df)
name = "fVf"
df = pd.read_csv("results_fVf.csv")
plot(name, df)

plt.legend()
plt.savefig(f"plot_l2.pdf", bbox_inches="tight")