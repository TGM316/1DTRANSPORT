from mpi4py import MPI
import numpy as np

def save_state_parallel(comm, n, u, ut, filename="./stateBackup/simulation_state.npz"):
    rank = comm.Get_rank()
    u_array = u.vector.gather_on_root()
    ut_array = ut.vector.gather_on_root()
    if rank == 0:
        np.savez(filename, n=n, u=u_array, ut=ut_array)

def load_state_parallel(comm, u, ut, filename="./stateBackup/simulation_state.npz"):
    rank = comm.Get_rank()
    if rank == 0:
        data = np.load(filename)
        n = int(data['n'])
        u_array = data['u']
        ut_array = data['ut']
    else:
        n = None
        u_array = None
        ut_array = None
    n = comm.bcast(n, root=0)
    u_array = comm.bcast(u_array, root=0)
    ut_array = comm.bcast(ut_array, root=0)
    # Diagnostics: print shape and checksum
    print(f"[Rank {rank}] Loaded n={n}, u_array.shape={np.shape(u_array)}, ut_array.shape={np.shape(ut_array)}, u_checksum={np.sum(u_array)}, ut_checksum={np.sum(ut_array)}")
    u.vector.setArray(u_array)
    ut.vector.setArray(ut_array)
    # Update ghost values after loading
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    ut.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    comm.barrier()
    return n
#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""
@author: Haifeng Wang (haifeng.wang@rub.de)
"""
import os
import tempfile
import zipfile
import shutil
import sys

import numpy as np
import math

from mpi4py import MPI
from petsc4py import PETSc

import ufl
from ufl.operators import sqrt
from ufl import (dot, grad, as_vector, variable, VectorElement)
from dolfinx import mesh, fem #, io, plot, log
from dolfinx.fem import (Constant, Function, FunctionSpace, VectorFunctionSpace)
from dolfinx.mesh import locate_entities
from dolfinx.io import XDMFFile

#########################################################
def remove_recreate_folder(folder_name):
    current_dir = os.getcwd()
    folder_path = os.path.join(current_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
        shutil.rmtree(folder_path)  #To avoid HD5-issue!
        print(f"Folder '{folder_name}' removed.")
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' recreated.")
#########################################################
def create_folder_if_not_exists(folder_name):
    current_dir = os.getcwd()
    folder_path = os.path.join(current_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created.")

#########################################################
# Function to save simulation state
def save_state(step, u, un, filename="./stateBackup/simulation_state.npz"):
    # Convert to numpy arrays
    u_array = u.vector[:]
    un_array = un.vector[:]
    # Save using np.savez
    np.savez(filename, step=step, u=u_array, un=un_array)

def load_state(u, un, filename="./stateBackup/simulation_state.npz"):
    # Load the data
    with np.load(filename, allow_pickle=False) as data:
        step = data['step']
        u.vector.setArray(data['u'])
        un.vector.setArray(data['un'])
    return step

# MPI-VERSION:
def save_state_mpi(comm, step, u, un, filename="./stateBackup/simulation_state.npz"):
    rank = comm.Get_rank()
    # Gather data from all processes to the root process
    u_local_array = u.vector.gather_on_root()
    un_local_array = un.vector.gather_on_root()

    if rank == 0:
        with tempfile.NamedTemporaryFile('wb', delete=False) as tmpfile:
            np.savez(tmpfile.name, step=step, u=u_local_array, un=un_local_array)
            shutil.move(tmpfile.name, filename)

#########################################################
tolCoord = 1e-5  # Tolerance for coordinate comparison
#########################################################
class GaussianHill:
    def __init__(self):
        self.t = 0.0
    def eval(self, x):
        x0tmp = 0.3
        Ltmp = x0tmp / 10.0  
        Btmp = 1.35e-12 * 95.0  #[mol/mm^3]
        return np.full(x.shape[1], Btmp * np.exp(-((x[1] - x0tmp) / Ltmp) ** 2))

##################################################################
class DeltaPulse:
    def __init__(self):
        self.t = 0.0  # Time attribute which might be used later
    def eval(self, x):
        x0tmp = 0.5
        rlTmp = 5
        pO2C = 1.35E-12  #pO2[mmHg] = C[mol/mm^3] / alpha[(mol/mm^3)/mmHg]; C=pO2*alpha
        coefTmp = 50 * pO2C /rlTmp  #pO2=C/alpha; C=pO2*alpha
        std_dev = 0.01 /rlTmp
        return np.full(x.shape[1], coefTmp * np.exp(-((x[1] - x0tmp) ** 2) / (2 * std_dev ** 2)))   
        
    
#########################################################
def assign_local_property_vertexBased(domain, value, V):
    kW = fem.Function(V)
    
    # Get the total number of vertices in the mesh
    num_vertices = domain.topology.index_map(0).size_local + domain.topology.index_map(0).num_ghosts
    vertex_coords = domain.geometry.x
    
    # Initialize an array to hold the new values
    kW_values = np.full(num_vertices, value, dtype=PETSc.ScalarType)
    
    # Iterate over all vertex coordinates
    for vertexID in range(num_vertices):
        # Check the z-coordinate of each vertex
        if vertex_coords[vertexID, 2] > 0:
            kW_values[vertexID] = 0.0
    
    # Get the dofmap to relate vertices to DoFs for CG-1 space
    dofmap = V.dofmap.list.array
    
    # Directly assign the modified values to the function's vector
    with kW.vector.localForm() as loc:
        loc.setValues(dofmap, kW_values[dofmap])

    # Update ghost values
    kW.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    return kW

#########################################################
#########################################################
def find_vertices_directly_connected_to_vertexi(vertex_index, mesh):
    tdim = mesh.topology.dim  # Topological dimension of the mesh
    mesh.topology.create_connectivity(0, tdim)  # Ensure vertex-to-cell connectivity is created
    mesh.topology.create_connectivity(tdim, 0)  # Ensure cell-to-vertex connectivity is created
    v_to_c = mesh.topology.connectivity(0, tdim)  # Vertex-to-cell connectivity
    c_to_v = mesh.topology.connectivity(tdim, 0)  # Cell-to-vertex connectivity

    connected_vertices = set()
    for cell in v_to_c.links(vertex_index):  # Cells connected to the vertex
        for v in c_to_v.links(cell):  # Vertices connected to the cell
            if v != vertex_index:
                connected_vertices.add(v)
    return list(connected_vertices)

#########################################################
def create_solver(target_space, petsc_options=None):
    # Create a dummy variational problem to initialize matrix and solver
    v = ufl.TestFunction(target_space)
    u = ufl.TrialFunction(target_space)
    dx = ufl.Measure('dx', domain=target_space.mesh)
    a_proj_dummy = ufl.inner(u, v) * dx

    # Wrap the form with dolfinx.fem.Form and assemble the matrix
    a_form = fem.form(a_proj_dummy)
    A = fem.petsc.assemble_matrix(a_form)
    A.assemble()

    # Create the solver
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    if petsc_options:
        for key, value in petsc_options.items():
            PETSc.Options().setValue(key, value)
    solver.setFromOptions()
    solver.setOperators(A)

    return solver, A


def project_function(target_space, source_function, solver, A):
    # Define the linear form for projection
    v = ufl.TestFunction(target_space)
    L = ufl.inner(source_function, v) * ufl.dx

    # Wrap the linear form with dolfinx.fem.Form
    L_form = fem.form(L)

    # Assemble the RHS vector
    b = fem.petsc.assemble_vector(L_form)

    # Update ghost values
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Create a function to store the solution
    target_function = fem.Function(target_space)

    # Solve the linear system
    solver.solve(b, target_function.vector)

    return target_function

##################
def cellDirVec_DG(domain, dirVect_cells):
    # Define a vector DG function space
    DG_space = VectorFunctionSpace(domain, ("DG", 0), dim=3)
    
    # Create a function to hold our piecewise direction vectors
    vec_function = Function(DG_space)
    
    # Flatten the dirVect_cells array to set the values in the vector
    flat_values = np.array(dirVect_cells).flatten()
    vec_function.vector.setArray(flat_values)
    
    if MPI.COMM_WORLD.Get_size() > 1:
        vec_function.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    return vec_function
    
#########################################################
def compute_directional_vectors_cells(domain):
    # Get the existing connectivity from cells to vertices
    c_to_v = domain.topology.connectivity(1, 0)
    #print(f"--> c_to_v: {c_to_v}")
    directional_vectors = np.zeros((domain.topology.index_map(1).size_local, 3))  #Initialized as [0,0,0]
    for cell_index in range(domain.topology.index_map(1).size_local):
        # Get vertex indices for the cell
        vertices_indices = c_to_v.links(cell_index)
        # Retrieve the coordinates of the vertices
        vertex_coords = np.array([domain.geometry.x[vertex_index] for vertex_index in vertices_indices])
        # Ensure we have two vertices for a line cell
        if vertex_coords.shape[0] != 2:
            raise RuntimeError(f"Cell {cell_index} does not have two vertices, found {vertex_coords.shape[0]}")
        # Calculate the directional vector
        vector = vertex_coords[1, :] - vertex_coords[0, :]
        norm = np.linalg.norm(vector)
        if norm != 0:
            directional_vectors[cell_index, :] = vector / norm
        else:
            # Handle the zero vector case
            directional_vectors[cell_index, :] = np.array([0, 0, 0])
            print("--> norm = 0 in calling compute_directional_vectors !")
            exit(0)
        #print(f"--> norm:{norm}; vector:{vector}; directional_vectors:{np.linalg.norm(directional_vectors[0])}")

    # Iterate through the direction vectors and print the ones that are different from all previous ones
    #if MPI.COMM_WORLD.Get_size() == 1:
    #    for i, dir_vec in enumerate(directional_vectors):
    #        if i == 0 or is_different_from_all(dir_vec, directional_vectors[:i]):
    #            print(f"--> Cell {i}: Direction Vector = {dir_vec}")

    return directional_vectors

#########################################################
def compute_length(inlet_coords, outlet_coords):
    inlet_coords = np.squeeze(np.asarray(inlet_coords))
    outlet_coords = np.squeeze(np.asarray(outlet_coords))
    # Calculate the direction vector
    direction_vector = outlet_coords - inlet_coords
    # Calculate the length of the direction vector
    return np.sqrt(np.dot(direction_vector, direction_vector))

#########################################################
def SHb_DB2001(domain, CF, solCoef):
    solCoef = Constant(domain, solCoef) 
    
    a1 = Constant(domain, 0.01524)
    a2 = Constant(domain, 2.7e-6)
    a3 = Constant(domain, 0.0)
    a4 = Constant(domain, 2.7e-6)
    
    alphaCF = CF / solCoef
    
    one = Constant(domain, 1.0)
    two = Constant(domain, 2.0)
    three = Constant(domain, 3.0)
    four = Constant(domain, 4.0)
    
    return (a1 * alphaCF + two * a2 * alphaCF ** two + three * a3 * alphaCF ** three + four * a4 * alphaCF ** four) / \
           (four * (one + a1 * alphaCF + a2 * alphaCF ** two + a3 * alphaCF ** three + a4 * alphaCF ** four))

#########################################################
def SHb(domain, CF, solCoef):
    a, b, c, n = 0.34332, 0.64073, 0.34128, 1.58678  # Fitted over range 0<=SHb<=1
    solCoef = Constant(domain, solCoef)  # [(mol/mm^3) /mmHg], keeps pO2 in mmHg!!!

    P50 = 26.8  # [mmHg] Half-saturation pressure of hemoglobin [ref: Dash2006]
    Temp= 37.0  # Temperature in degrees Celsius
    pH = 7.4    # pH value
    Pco2 = 40.0  # [mmHg] Partial pressure of CO2 in mmHg
    T_pH_Pco2 = 10 ** (0.024 * (37.0 - Temp) + 0.4 * (pH - 7.4) + 0.06 * np.log(40.0 / Pco2))
    
    pO2 = CF / solCoef  # [mmHg]
    xTmp = T_pH_Pco2 * pO2 / P50
    SHb = (a * xTmp ** n + b * xTmp ** (2 * n)) / (1 + c * xTmp ** n + b * xTmp ** (2 * n))
    return SHb
    
#########################################################
if __name__ == "__main__":
    print('--> auxiliaryFunctions!')
#########################################################
