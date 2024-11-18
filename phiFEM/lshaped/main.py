from basix.ufl import element
import dolfinx as dfx
from dolfinx.io import XDMFFile
import jax.numpy as jnp
from mpi4py import MPI
import numpy as np
import os
from petsc4py import PETSc
from utils.solver import PhiFEMSolver
from utils.continuous_functions import Levelset, ContinuousFunction
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
        mesh_type = "Submesh"
    else:
        output_dir = os.path.join(parent_dir, "output", "bg_mesh", ref_method)
        mesh_type = "Bg mesh"

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    tilt_angle = np.pi/6. + np.pi/2.
    shift = jnp.array([jnp.pi/32., jnp.pi/32.])

    def rotate(angle, x):
        # Rotation matrix
        R = np.array([[ np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
        rotated = R.dot(np.array(x))
        return rotated

    def line(x, y, a, b, c):
        rotated = rotate(tilt_angle, [x, y])
        return a*rotated[0] + b*rotated[1] + np.full_like(x, c)

    def expression_levelset(x, y):
        x_shift = x - np.full_like(x, shift[0])
        y_shift = y - np.full_like(y, shift[1])

        line_1 = line(x_shift, y_shift, -1.,  0.,   0.)
        line_2 = line(x_shift, y_shift,  0., -1.,   0.)
        line_3 = line(x_shift, y_shift,  1.,  0., -0.5)
        line_4 = line(x_shift, y_shift,  0.,  1., -0.5)
        line_5 = line(x_shift, y_shift,  0., -1., -0.5)
        line_6 = line(x_shift, y_shift, -1.,  0., -0.5)

        reentrant_corner = np.minimum(line_1, line_2)
        top_right_corner = np.maximum(line_3, line_4)
        corner           = np.maximum(reentrant_corner, top_right_corner)
        horizontal_leg   = np.maximum(corner, line_5)
        vertical_leg     = np.maximum(horizontal_leg, line_6)
        return vertical_leg
    
    def expression_rhs(x, y):
        return np.ones_like(x)

    phi = Levelset(expression_levelset)
    f = ContinuousFunction(expression_rhs)

    """
    Read bg_mesh
    """
    bg_mesh_path_xdmf = os.path.join(parent_dir, "square.xdmf")
    bg_mesh_path_h5   = os.path.join(parent_dir, "square.h5")

    if (not os.path.isfile(bg_mesh_path_h5)) or (not os.path.isfile(bg_mesh_path_xdmf)):
        cprint(f"{bg_mesh_path_h5} or {bg_mesh_path_xdmf} not found, we create them.", print_save)
        bg_mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)

        # Shift and scale the bg_mesh
        bg_mesh.geometry.x[:, 0] -= 0.5
        bg_mesh.geometry.x[:, 1] -= 0.5
        bg_mesh.geometry.x[:, 0] *= 2.
        bg_mesh.geometry.x[:, 1] *= 2.

        with XDMFFile(bg_mesh.comm, "./square.xdmf", "w") as of:
            of.write_mesh(bg_mesh)
    else:
        with XDMFFile(MPI.COMM_WORLD, "./square.xdmf", "r") as fi:
            bg_mesh = fi.read_mesh()

    data = ["dofs", "H10 estimator"]
    if print_save:
        results_saver = ResultsSaver(output_dir, data)

    working_mesh = bg_mesh
    for i in range(max_it):
        cprint(f"Mesh: {mesh_type}. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Creation FE space and data interpolation", print_save)
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
        cprint(f"Mesh: {mesh_type}. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Data interpolation.", print_save)
        phiFEM_solver.set_data(f, phi)
        cprint(f"Mesh: {mesh_type}. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Mesh tags computation.", print_save)
        phiFEM_solver.compute_tags(create_submesh=compute_submesh, padding=True)
        cprint(f"Mesh: {mesh_type}. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Variational formulation set up.", print_save)
        v0, dx, dS, num_dofs = phiFEM_solver.set_variational_formulation()
        cprint(f"Mesh: {mesh_type}. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Linear system assembly.", print_save)
        phiFEM_solver.assemble()
        cprint(f"Mesh: {mesh_type}. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Solve.", print_save)
        phiFEM_solver.solve()

        if compute_submesh:
            uh = phiFEM_solver.compute_submesh_solution()
            working_mesh = phiFEM_solver.submesh
        else:
            uh = phiFEM_solver.compute_bg_mesh_solution()
            working_mesh = phiFEM_solver.bg_mesh

        cprint(f"Mesh: {mesh_type}. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Ouput save and error computation.", print_save)
        
        phiV = phi.interpolate(phiFEM_solver.levelset_space)
        fV = f.interpolate(phiFEM_solver.FE_space)

        cprint(f"Mesh: {mesh_type}. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: a posteriori error estimation.", print_save)
        eta_h = estimate_residual(fV, phiFEM_solver)
        h10_est = np.sqrt(sum(eta_h.x.array[:]))

        # Save results
        if print_save:
            results_saver.save_function(eta_h, f"eta_h_{str(i).zfill(2)}")
            results_saver.save_function(phiV,  f"phi_V_{str(i).zfill(2)}")
            if not compute_submesh:
                results_saver.save_function(v0, f"v0_{str(i).zfill(2)}")
            results_saver.save_function(uh, f"uh_{str(i).zfill(2)}")
            results_saver.save_values([num_dofs,
                                       h10_est],
                                       prnt=True)

        # Marking
        if i < max_it - 1:
            cprint(f"Mesh: {mesh_type}. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Mesh refinement.", print_save)
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
                assert not compute_submesh, "Submesh has been created, we cannot refine the background mesh."
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
                      ref_method="omega_h",
                      compute_submesh=False)
    poisson_dirichlet(10,
                      7,
                      print_save=True,
                      ref_method="omega_h",
                      compute_submesh=True)