from dolfinx.mesh import Mesh, MeshTags
import numpy.typing as npt 
from phifem.continuous_functions import Levelset
from typing import Any, Tuple

# TODO: Modify to use in parallel

def _select_entities(mesh: Mesh, levelset: Levelset, edim: int, padding: bool =False) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any] | list[int]]: ...

def tag_entities(mesh: Mesh,
                 levelset: Levelset,
                 edim: int,
                 cells_tags: MeshTags | None = None,
                 padding: bool =False,
                 plot: bool =False) -> MeshTags: ...