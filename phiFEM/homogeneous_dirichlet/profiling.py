from main import poisson_dirichlet
import cProfile
import os
import pstats

parent_dir = os.path.dirname(__file__)

if __name__=="__main__":
    os.remove(os.path.join(parent_dir, "square.h5"))
    os.remove(os.path.join(parent_dir, "square.xdmf"))
    profiler = cProfile.Profile()
    profiler.enable()
    poisson_dirichlet(1000,
                      1,
                      ref_method="omega_h",
                      compute_submesh=False)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("ncalls")
    stats.strip_dirs()
    stats.dump_stats("profile_bg_mesh.prof")
    os.remove(os.path.join(parent_dir, "square.h5"))
    os.remove(os.path.join(parent_dir, "square.xdmf"))
    profiler = cProfile.Profile()
    profiler.enable()
    poisson_dirichlet(1000,
                      1,
                      ref_method="omega_h",
                      compute_submesh=True)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("ncalls")
    stats.strip_dirs()
    stats.dump_stats("profile_submesh.prof")