from dolfinx.mesh import Mesh, MeshTags
from dolfinx.fem import Function

def tag_cells(mesh: Mesh,
              discrete_levelset: Function,
              plot: bool =False) -> MeshTags: ...
        
def tag_facets(mesh: Mesh,
               discrete_levelset: Function,
               cells_tags: MeshTags,
               plot: bool =False) -> MeshTags: ...