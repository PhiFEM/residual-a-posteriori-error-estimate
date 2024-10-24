import basix as bx
from basix.ufl import element
from basix import CellType, ElementFamily
import dolfinx as dfx
from dolfinx.fem import assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector, set_bc
from dolfinx.io import XDMFFile, gmshio
import jax
import jax.numpy as jnp
from mpi4py import MPI
import numpy as np
import os
import pandas as pd
from petsc4py import PETSc
import ufl
from ufl import inner, jump, grad, div, dot, avg

from utils.compute_meshtags import tag_entities
from utils.classes import Levelset, ExactSolution

output_dir = "output"
ref_max = 5
angle = np.pi/6.
quadrature_degree = 4  # Quadrature degree for P1
sigma_D = 1.

def rotation(x):
    return (jnp.cos(angle)*x[0] + jnp.sin(angle)*x[1], -jnp.sin(angle)*x[0] + jnp.cos(angle)*x[1])

def expression_levelset(x, y):
    return rotation([x, y])[0]**100 + rotation([x, y])[1]**100 - 1

def u_exact(x):
    return np.sin(2. * np.pi * rotation(x)[0]) * np.sin(2. * np.pi * rotation(x)[1])

u_exact = ExactSolution(lambda x, y: jnp.sin(np.pi * rotation((x,y))[0]) * jnp.sin(np.pi * rotation((x,y))[1]))
f = u_exact.negative_laplacian()

if not os.path.isdir(output_dir):
	print(f"{output_dir} directory not found, we create it.")
	os.mkdir(os.path.join(".", output_dir))

phi = Levelset(expression_levelset)

"""
Read mesh
"""
print("Read mesh")
mesh, cell_tags, facet_tags = gmshio.read_from_msh("./background.msh", MPI.COMM_WORLD, gdim=2)

print("Creation FE space and data interpolation")
CG1Element = element("CG", mesh.topology.cell_name(), 1)
V = dfx.fem.functionspace(mesh, CG1Element)
phiV = dfx.fem.Function(V)
phiV.interpolate(phi.dolfinx_call)
fV = dfx.fem.Function(V)
fV.interpolate(f)

print("Compute entities tags")
cells_tags  = tag_entities(mesh, phi, mesh.topology.dim, )
facets_tags = tag_entities(mesh, phi, mesh.topology.dim - 1)

print("Define measures and set up FE problem")
DG0Element = element("DG", mesh.topology.cell_name(), 0)
V0 = dfx.fem.functionspace(mesh, DG0Element)
v0 = dfx.fem.Function(V0)
v0.vector.set(0.)
v0.vector.array[cells_tags.indices[np.where(cells_tags.values == 1)]] = 1.

dx = ufl.Measure("dx",
                    domain=mesh,
                    subdomain_data=cells_tags,
                    metadata={"quadrature_degree": quadrature_degree})

dS = ufl.Measure("dS",
                domain=mesh,
                subdomain_data=facets_tags,
                metadata={"quadrature_degree": quadrature_degree})


h = ufl.CellDiameter(mesh)
n = ufl.FacetNormal(mesh)

w = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# This term is useless (always zero, everybody hate it)
# but PETSc complains if I don't include it
useless = inner(w, v) * v0 * dx(3)

a = inner(grad(phiV * w), grad(phiV * v)) * dx(1) + inner(grad(phiV * w), grad(phiV * v)) * dx(2) \
    - inner(jump(grad(phiV * w), grad(phiV)), jump(phiV * v * v0)) * dS(4) \
    + sigma_D * avg(h) * inner(jump(grad(phiV * w), n), jump(grad(phiV * v), n)) * dS(2) \
    + sigma_D * h**2 * inner(div(grad(phiV * w)), div(grad(phiV * v))) * dx(2) \
    + useless

L = inner(fV, phiV * v) * dx(1) + inner(fV, phiV * v) * dx(2) \
    - sigma_D * h**2 * inner(fV, div(grad(phiV * v))) * dx(2)

wh = dfx.fem.Function(V)
wh.vector.set(0.0)

a_form = dfx.fem.form(a)
b_form = dfx.fem.form(L)
print("Matrix and RHS assembly")
A = assemble_matrix(a_form)
A.assemble()
b = assemble_vector(b_form)

# Parametrization of the PETSc solver
options = PETSc.Options()
options["ksp_type"] = "cg"
options["pc_type"] = "hypre"
options["ksp_rtol"] = 1e-7
options["pc_hypre_type"] = "boomeramg"
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOperators(A)
solver.setFromOptions()

# Solve
print("Solve")
solver.solve(b, wh.vector)

print("Interpolation exact solution and save output")
u_exact_V = dfx.fem.Function(V)
u_exact_V.interpolate(u_exact.dolfinx_call)

uh = dfx.fem.Function(V)
uh.vector.array = wh.vector.array * phiV.vector.array

eh = dfx.fem.Function(V)
eh.vector.array = uh.vector.array - u_exact_V.vector.array

with XDMFFile(mesh.comm, f"output/uh.xdmf", "w") as of:
    of.write_mesh(mesh)
    of.write_function(uh)

with XDMFFile(mesh.comm, f"output/u_exact_V.xdmf", "w") as of:
    of.write_mesh(mesh)
    of.write_function(u_exact_V)

with XDMFFile(mesh.comm, f"output/phi_V.xdmf", "w") as of:
    of.write_mesh(mesh)
    of.write_function(phiV)

with XDMFFile(mesh.comm, f"output/eh.xdmf", "w") as of:
    of.write_mesh(mesh)
    of.write_function(eh)