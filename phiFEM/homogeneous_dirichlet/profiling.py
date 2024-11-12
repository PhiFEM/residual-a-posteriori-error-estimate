from main import poisson_dirichlet
import cProfile
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pstats
import io

parent_dir = os.path.dirname(__file__)


def _prof_to_csv(prof: cProfile.Profile):
    '''
    prof_to_csv from here: https://gist.github.com/ralfstx/a173a7e4c37afa105a66f371a09aa83e
    '''
    out_stream = io.StringIO()
    pstats.Stats(prof, stream=out_stream).print_stats()
    result = out_stream.getvalue()
    # chop off header lines
    result = 'ncalls' + result.split('ncalls')[-1]
    lines = [','.join(line.rstrip().split(None, 5)) for line in result.split('\n')]
    return '\n'.join(lines)

def _profile(filename, args_dict):
    mesh_path_h5   = os.path.join(parent_dir, "square.h5")
    mesh_path_xdmf = os.path.join(parent_dir, "square.xdmf")
    # If mesh exists, we remove it to regenerate the proper one used to profile
    if (os.path.isfile(mesh_path_h5)):
        os.remove(mesh_path_h5)
    if (os.path.isfile(mesh_path_xdmf)):
        os.remove(mesh_path_xdmf)

    profiler = cProfile.Profile()
    profiler.enable()
    poisson_dirichlet(**args_dict)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.strip_dirs()
    stats.dump_stats(os.path.join(parent_dir, filename + ".prof"))
    
    txtfile = _prof_to_csv(profiler)

    with open(filename.rsplit('.')[0]+'.csv','w') as of:
        of.write(txtfile)

    cumtime_dict = {}
    df = pd.read_csv(io.StringIO(txtfile), sep=",")
    classes_df = df[df["filename:lineno(function)"].str.contains("classes.py")]
    keys = ["compute_tags", "set_variational_formulation", "assemble", "solve"]
    for key in keys:
        cumtime_dict[key] = classes_df[classes_df["filename:lineno(function)"].str.contains(key)]["cumtime"].values[0]

    cumtime_df = pd.DataFrame([cumtime_dict])
    cumtime_hist_file = os.path.join(parent_dir, "cumtime_hist.csv")
    if os.path.isfile(cumtime_hist_file):
        df = pd.read_csv(cumtime_hist_file)
        df = pd.concat([df, cumtime_df], ignore_index=True)
    else:
        df = cumtime_df
    
    fig = plt.figure()
    xs = np.arange(len(df[keys[0]].values))
    for key in keys:
        vals = df[key].values[-10:]
        mean = np.mean(vals[:-1])
        plt.plot(xs[-10:], vals, label=key)
        plt.plot(xs[-10:], np.full_like(xs[-10:], mean, dtype=np.double), "--", color="black")
    fig.legend(loc='outside upper center', bbox_to_anchor=(0.5, 1.), ncol=4)
    plt.ylabel("Cumtime (s)")
    plt.xlabel("Run n°")
    plt.savefig(os.path.join(parent_dir, "cumtime_hist_plot.pdf"), bbox_inches="tight")

    df.to_csv(cumtime_hist_file, index=False)

    # We remove the profiling mesh
    os.remove(mesh_path_h5)
    os.remove(mesh_path_xdmf)

if __name__=="__main__":
    args_dict = {"N": 1000, "max_it": 1, "ref_method": "omega_h", "compute_submesh": True}
    _profile("profile", args_dict)