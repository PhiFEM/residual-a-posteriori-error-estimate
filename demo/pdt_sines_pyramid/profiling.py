import cProfile
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import pandas as pd
import pstats
import io
from phiFEM.phifem.poisson_dirichlet import poisson_dirichlet_phiFEM

parent_dir = os.path.split(os.path.abspath(__file__))[0]

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
    profiler = cProfile.Profile()
    profiler.enable()
    poisson_dirichlet_phiFEM(**args_dict)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.strip_dirs()
    stats.dump_stats(os.path.join(parent_dir, filename + ".prof"))
    
    txtfile = _prof_to_csv(profiler)

    with open(filename.rsplit('.')[0]+'.csv','w') as of:
        of.write(txtfile)

    cumtime_dict = {}
    df = pd.read_csv(io.StringIO(txtfile), sep=",")
    classes_df = df[df["filename:lineno(function)"].str.contains("solver.py")]
    keys = ["compute_tags", "set_variational_formulation", "assemble", "solve", "estimate"]
    for key in keys:
        cumtime_dict[key] = classes_df[classes_df["filename:lineno(function)"].str.contains(key)]["cumtime"].values[0]
        print(key)
        print(cumtime_dict[key])

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
        num_points = min(20, len(df[key].values))
        vals = df[key].values[-num_points:]
        mean = np.mean(vals[:-1])
        plt.semilogy(xs[-num_points:], vals, label=key)
        plt.semilogy(xs[-num_points:], np.full_like(xs[-num_points:], mean, dtype=np.double), "--", color="black")
    fig.legend(loc='outside upper center', bbox_to_anchor=(0.5, 1.), ncol=4)
    plt.ylabel("Cumtime (s)")
    plt.xlabel("Run n°")
    plt.savefig(os.path.join(parent_dir, "cumtime_hist_plot.pdf"), bbox_inches="tight")

    df.to_csv(cumtime_hist_file, index=False)

if __name__=="__main__":
    tilt_angle = np.pi/6.
    def rotation(angle, x):
        R = jnp.array([[ jnp.cos(angle), jnp.sin(angle)],
                       [-jnp.sin(angle), jnp.cos(angle)]])
        return R.dot(jnp.asarray(x))

    def expression_levelset(x, y):
        def fct(x, y):
            return jnp.sum(jnp.abs(rotation(tilt_angle - jnp.pi/4., [x, y])), axis=0)
        return fct(x, y) - np.sqrt(2.)/2.

    def expression_u_exact(x, y):
        return jnp.sin(2. * jnp.pi * rotation(tilt_angle, [x, y])[0]) * jnp.sin(2. * jnp.pi * rotation(tilt_angle, [x, y])[1])
    
    args_dict = {"cl":                  0.1,
                "max_it":               1,
                "expression_levelset":  expression_levelset,
                "source_dir":           parent_dir,
                "expression_rhs":       None,
                "expression_u_exact":   expression_u_exact,
                "bg_mesh_corners":      [np.array([0., 0.]),
                                         np.array([1., 1.])],
                "quadrature_degree":    4,
                "sigma_D":              1.,
                "save_output":          False,
                "ref_method":           "adaptive"}
    _profile("profile", args_dict)