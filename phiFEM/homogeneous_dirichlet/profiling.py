from main import poisson_dirichlet
import cProfile, pstats

if __name__=="__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    poisson_dirichlet(500, 1)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("ncalls")
    stats.strip_dirs()
    stats.dump_stats("profile.prof")