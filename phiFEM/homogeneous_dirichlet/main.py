from basix.ufl import element
import dolfinx as dfx
from dolfinx.fem import assemble_vector
from dolfinx.io import XDMFFile
import jax.numpy as jnp
from mpi4py import MPI
import numpy as np
import os
from petsc4py import PETSc
import ufl
from ufl import inner, grad
from utils.solver import PhiFEMSolver
from utils.continuous_functions import Levelset, ExactSolution
from utils.saver import ResultsSaver
from utils.mesh_scripts import compute_facets_to_refine
from utils.estimation import estimate_residual, marking

def cprint(str2print, print_save=True, file=None):
    if print_save:
        print(str2print, file=file)

def poisson_dirichlet(N,
                      max_it,
                      quadrature_degree=4,
                      sigma_D=1.,
                      print_save=False,
                      ref_method="background",
                      compute_submesh=False):
    parent_dir = os.path.dirname(__file__)
    if compute_submesh:
        output_dir = os.path.join(parent_dir, "output", "submesh", ref_method)
    else:
        output_dir = os.path.join(parent_dir, "output", "bg_mesh", ref_method)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    tilt_angle = np.pi/6.

    def rotation(angle, x):
        R = jnp.array([[ jnp.cos(angle), jnp.sin(angle)],
                       [-jnp.sin(angle), jnp.cos(angle)]])
        return R.dot(jnp.asarray(x))

    # Defines a tilted square
    def expression_levelset(x, y):
        def fct(x, y):
            return jnp.sum(jnp.abs(rotation(tilt_angle - jnp.pi/4., [x, y])), axis=0)
        return fct(x, y) - np.sqrt(2.)/2.

    def expression_u_exact(x, y):
        return jnp.sin(2. * jnp.pi * rotation(tilt_angle, [x, y])[0]) * jnp.sin(2. * jnp.pi * rotation(tilt_angle, [x, y])[1])

    phi = Levelset(expression_levelset)
    u_exact = ExactSolution(expression_u_exact)
    u_exact.compute_negative_laplacian()
    f = u_exact.nlap

    """
    Read mesh
    """
    mesh_path_xdmf = os.path.join(parent_dir, "square.xdmf")
    mesh_path_h5   = os.path.join(parent_dir, "square.h5")

    if (not os.path.isfile(mesh_path_h5)) or (not os.path.isfile(mesh_path_xdmf)):
        cprint(f"{mesh_path_h5} or {mesh_path_xdmf} not found, we create them.", print_save)
        bg_mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)

        # Shift and scale the mesh
        bg_mesh.geometry.x[:, 0] -= 0.5
        bg_mesh.geometry.x[:, 1] -= 0.5
        bg_mesh.geometry.x[:, 0] *= 2.
        bg_mesh.geometry.x[:, 1] *= 2.

        with XDMFFile(bg_mesh.comm, "./square.xdmf", "w") as of:
            of.write_mesh(bg_mesh)
    else:
        with XDMFFile(MPI.COMM_WORLD, "./square.xdmf", "r") as fi:
            bg_mesh = fi.read_mesh()

    data = ["dofs", "H10 error", "L2 error", "H10 estimator"]
    if print_save:
        results_saver = ResultsSaver(output_dir, data)

    working_mesh = bg_mesh
    for i in range(max_it):
        cprint(f"Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Creation FE space and data interpolation", print_save)
        CG1Element = element("CG", working_mesh.topology.cell_name(), 1)

        # Parametrization of the PETSc solver
        options = PETSc.Options()
        options["ksp_type"] = "cg"
        options["pc_type"] = "hypre"
        options["ksp_rtol"] = 1e-7
        options["pc_hypre_type"] = "boomeramg"
        petsc_solver = PETSc.KSP().create(working_mesh.comm)
        petsc_solver.setFromOptions()

        phiFEM_solver = PhiFEMSolver(working_mesh, CG1Element, petsc_solver, num_step=i)
        cprint(f"Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Data interpolation.", print_save)
        phiFEM_solver.set_data(f, phi)
        cprint(f"Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Mesh tags computation.", print_save)
        phiFEM_solver.compute_tags(create_submesh=compute_submesh, padding=True)
        cprint(f"Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Variational formulation set up.", print_save)
        v0, dx, dS, num_dofs = phiFEM_solver.set_variational_formulation()
        cprint(f"Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Linear system assembly.", print_save)
        phiFEM_solver.assemble()
        cprint(f"Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Solve.", print_save)
        phiFEM_solver.solve()

        if compute_submesh:
            uh = phiFEM_solver.compute_submesh_solution()
            working_mesh = phiFEM_solver.submesh
            working_cells_tags = phiFEM_solver.submesh_cells_tags
        else:
            uh = phiFEM_solver.compute_bg_mesh_solution()
            working_mesh = phiFEM_solver.bg_mesh
            working_cells_tags = phiFEM_solver.bg_mesh_cells_tags

        cprint(f"Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Ouput save and error computation.", print_save)
        
        CG2Element = element("CG", working_mesh.topology.cell_name(), 2)
        V = dfx.fem.functionspace(working_mesh, CG1Element)
        V2 = dfx.fem.functionspace(working_mesh, CG2Element)

        phiV = phi.interpolate(V)
        fV = f.interpolate(V)
        u_exact_V = u_exact.interpolate(V)
        u_exact_V2 = u_exact.interpolate(V2)
        uh_V2 = dfx.fem.Function(V2)
        uh_V2.interpolate(uh)
        e_V2 = dfx.fem.Function(V2)
        e_V2.x.array[:] = u_exact_V2.x.array - uh_V2.x.array

        dx2 = ufl.Measure("dx",
                          domain=working_mesh,
                          subdomain_data=working_cells_tags,
                          metadata={"quadrature_degree": 6})

        DG0Element = element("DG", working_mesh.topology.cell_name(), 0)
        V0 = dfx.fem.functionspace(working_mesh, DG0Element)
        w0 = ufl.TrialFunction(V0)

        L2_norm_local = inner(inner(e_V2, e_V2), w0) * v0 * dx2(1) + inner(inner(e_V2, e_V2), w0) * v0 * dx2(2)
        H10_norm_local = inner(inner(grad(e_V2), grad(e_V2)), w0) * v0 * dx2(1) + inner(inner(grad(e_V2), grad(e_V2)), w0) * v0 * dx2(2)

        L2_error_0 = dfx.fem.Function(V0)
        H10_error_0 = dfx.fem.Function(V0)

        L2_norm_local_form = dfx.fem.form(L2_norm_local)
        L2_error_0.x.array[:] = assemble_vector(L2_norm_local_form).array
        L2_error_global = np.sqrt(sum(L2_error_0.x.array[:]))

        H10_norm_local_form = dfx.fem.form(H10_norm_local)
        H10_error_0.x.array[:] = assemble_vector(H10_norm_local_form).array
        H10_error_global = np.sqrt(sum(H10_error_0.x.array[:]))

        cprint(f"Method: {ref_method}. Iteration n° {str(i).zfill(2)}: a posteriori error estimation.", print_save)
        eta_h = estimate_residual(fV, phiFEM_solver, V0=V0)
        h10_est = np.sqrt(sum(eta_h.x.array[:]))

        # Save results
        if print_save:
            results_saver.save_function(L2_error_0,  f"L2_error_{str(i).zfill(2)}")
            results_saver.save_function(H10_error_0, f"H10_error_{str(i).zfill(2)}")
            results_saver.save_function(u_exact_V,   f"u_exact_V_{str(i).zfill(2)}")
            results_saver.save_function(eta_h,       f"eta_h_{str(i).zfill(2)}")
            results_saver.save_function(phiV,        f"phi_V_{str(i).zfill(2)}")
            if not compute_submesh:
                results_saver.save_function(v0,      f"v0_{str(i).zfill(2)}")
            results_saver.save_function(uh,          f"uh_{str(i).zfill(2)}")
            results_saver.save_values([num_dofs,
                                       H10_error_global,
                                       L2_error_global,
                                       h10_est],
                                       prnt=True)

        # Marking
        if i < max_it - 1:
            cprint(f"Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Mesh refinement.", print_save)
            facets_tags = phiFEM_solver.facets_tags

            # Uniform refinement (Omega_h only)
            if ref_method == "omega_h":
                if compute_submesh:
                    working_mesh = dfx.mesh.refine(working_mesh)
                else:
                    facets_to_refine = compute_facets_to_refine(working_mesh, facets_tags)
                    working_mesh = dfx.mesh.refine(working_mesh, facets_to_refine)

            # Uniform refinement (background mesh)
            if ref_method == "background":
                cprint("Submesh has been created, we refine all the submesh.")
                working_mesh = dfx.mesh.refine(working_mesh)

            # Adaptive refinement
            if ref_method == "adaptive":
                facets2ref = marking(eta_h)
                working_mesh = dfx.mesh.refine(working_mesh, facets2ref)

        cprint("\n", print_save)

if __name__=="__main__":
    poisson_dirichlet(10,
                      25,
                      print_save=True,
                      ref_method="adaptive",
                      compute_submesh=False)
    poisson_dirichlet(10,
                      25,
                      print_save=True,
                      ref_method="adaptive",
                      compute_submesh=True)
    poisson_dirichlet(10,
                      7,
                      print_save=True,
                      ref_method="background",
                      compute_submesh=False)
    poisson_dirichlet(10,
                      7,
                      print_save=True,
                      ref_method="background",
                      compute_submesh=True)