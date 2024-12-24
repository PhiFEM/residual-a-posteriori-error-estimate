from collections.abc import Callable
from dolfinx.cpp.graph import AdjacencyList_int32
from dolfinx.mesh import Mesh, MeshTags
from dolfinx.fem import Function
from   matplotlib.axes import Axes
import numpy as np
import numpy.typing as npt
from   os import PathLike
from phifem.continuous_functions import Levelset
from typing import Tuple

NDArrayTuple = Tuple[npt.NDArray[np.float64], ...]
NDArrayFunction = Callable[[NDArrayTuple], npt.NDArray[np.float64]]

def mesh2d_from_levelset(lc: float,
                         levelset: Levelset,
                         level:float = 0.,
                         bbox: npt.NDArray[np.float64] = np.array([[-1., 1.], [-1., 1.]]),
                         geom_vertices: npt.NDArray[np.float64] | None =None,
                         output_dir: PathLike | None =None,
                         file_name: str ="conforming_mesh") -> npt.NDArray[np.float64]: ...

def compute_outward_normal(mesh: Mesh, levelset: Levelset) -> Function: ...

def reshape_facets_map(f2c_connect: AdjacencyList_int32) -> npt.NDArray[np.int32]: ...

def plot_mesh_tags(mesh: Mesh,
                   mesh_tags: MeshTags,
                   ax: Axes | None = None,
                   display_indices: bool = False,
                   expression_levelset: NDArrayFunction | None = None) -> Axes: ...