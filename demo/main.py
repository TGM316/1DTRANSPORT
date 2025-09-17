#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""
##############################
@author: Haifeng Wang (haifeng.wang@rub.de)
# DOLFINx version: 0.6.0.0
##############################
"""

##############################
# Import libraries:
import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt

import scipy.io as sio
from scipy.io import loadmat

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh, io, log  
from dolfinx.fem import (Constant, Function, FunctionSpace, VectorFunctionSpace, dirichletbc,
                         assemble_scalar, form, locate_dofs_topological, set_bc, locate_dofs_geometrical)
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, NonlinearProblem

from dolfinx.nls.petsc import NewtonSolver  
from dolfinx.io import (XDMFFile, VTXWriter, distribute_entity_data, gmshio)
from dolfinx.cpp.mesh import entities_to_geometry

from ufl import (SpatialCoordinate, TestFunction, TrialFunction, VectorElement, dx, ds, dS, dot, inner, jump, inv, div,
                 grad, nabla_grad, lhs, rhs, sym, as_vector, as_matrix, FacetNormal, CellDiameter, Measure,
                 FiniteElement, MixedElement, Identity, split, derivative, variable, conditional)
from ufl.operators import sqrt, exp

sys.path.append('../src')  # Adjust this path as necessary
from auxiliaryFunctions import *

import csv
from time import perf_counter

# Start the timer on all processes
startTime = perf_counter()

commMPI = MPI.COMM_WORLD
rankMPI = commMPI.Get_rank()
sizeMPI = commMPI.Get_size()

##################
# Note: 1M=1mol/L=1e-6mol/mm^3; 1M=1000mM (millimolar); 
#       1 mM = 1e-3 M = 1e-9 mol/mm^3;
#       1L=1e6mm^3=1000cm^3=1000mL; 1mL=1cm^3 = 1e3mm^3; 1cm=10mm;
#       1 mmHg = 133.322 g/(mm·s²);
# N=kg·m/s², Pa=N/m²; 1 mmHg≈133.322 Pa 
# 1 mmHg = 133.322 g/(mm·s²)=133.322 kg/(m·s²) = 133.322*1000 g/(1000mm·s²)
###### Base units: "mm-gram-second-mole"
Molar_to_mol_mm = 1E-6  # 1M = 1e-6mol/mm^3
mL_to_mm = 1000.0  # 1mL = 1e3mm^3 = 1cm^3
mmHg_to_mmGs = 133.322  # 1mmHg = 133.322g/(mm·s²)
######
# Solubility coefficient of oxygen (0.00135mM/mmHg; 1mM=1e-9mol/mm^3)
pO2C = 1.35E-12  # pO2[mmHg] = C[mol/mm^3] / alpha[(mol/mm^3)/mmHg]; C=pO2*alpha
constPO2inlet = 95  #TODO
##################
isNonPulsatileCase = 0  # TODO: set to '1' for non-pulsatile cases!
#########################################################
matDataPath = '../matlabData/results_cleaned_nonisch/'   
XDMFmeahFileName = '../mesh/1786v_shift_dL0.0005_2tags.xdmf'
folder_name = "1786v_nonischemic"
if MPI.COMM_WORLD.rank == 0:  # Only the root process creates the folder
    remove_recreate_folder(folder_name)

if MPI.COMM_WORLD.Get_size() > 1:
    MPI.COMM_WORLD.barrier()  # Synchronize processes after folder creation

##############################
###### Solver setting: [Speed: gmres+jacobi > preonly+jacobi > preonly+lu > fgmres+jacobi]
convergCriteria = "incremental"  # TODO: "incremental"; NOTE: "residual" did not work well!
kspType_ = "gmres"  # TODO: "preonly", "gmres", "fgmres", "bicg", "bicgstab", "cg", "minres", "mumps", "petsc", "default"
pcType_ = "jacobi"  # TODO: "jacobi", "lu", "ilu", "amg", "bjacobi", "none",
maxIter_ksp = 1000
##############################
######
###
# TODO: '0' for 'CB=0, CT=CF'; '1' for 'CB!=0, difDHb=difD'; '2' for 'CB!=0, difDHb=difD/65'; '3' for 'CB!=0, difDHb=0'
useCTCFCB = 2
###
useR22ForTissue = 1  # TODO: 0 for NOT using R22; 1 for using R22
wTissueDiffusion = 1  # TODO: 1 for including Diffusion; 0 for w/o Diffusion
benchmarkAdvDiff = 0  # TODO: set to '0' if not benchmark!
benchPermeation = 0  # TODO: set to '0' if not benchmark!
isSingleVessel = 0  # TODO: set to '0' if not using 1 vessel!
constQFlag = 0  # TODO: set to '0' if not constantQ to use matlab results! --> Same Q for all vessels!!!
constRFlag = 0  # TODO: set to '0' if not constantR to use matlab results! --> Same R for all vessels!!!

######
# Free O2 diffusion coefficient (cm^2/s=100mm^2/s): [0.95, 1.5, 2.1, 2.41]e-5cm^2/s
difD_value = 2.41e-5 * 100  # 'PeMax' needs a non-UFL value!
dtTmp = 0.001  # Better keep dtTmp == dt_MatLab!!!
dtAdjustedAutomatically = 0  #See below "if isNonPulsatileCase == 0 and dt > dtTmp:"
######
findMaxPeMaxCFL = 2  # TODO: Try '2' first, if not OK, use '1'
###
CFLreduceFactor = 10  # TODO: if '1.1 (CFL=0.9)' crashed, try '2 (CFL=0.5)' or '5(CFL=0.2)'!!!
if CFLreduceFactor > 10:
    CFLreduceFactor = 10

numSteps_test = 0  # TODO: set to '0' or '-1' if not used!
PeCritical = -1000  # Use SUPG when 'Pe > PeCritical'!
steadySUPG = 0
######
initMethod = 'initConstant'
######
##############################
##############################
ratioVtVb = 12.5  
kWratioTmp = 1.0 #*5/3.5
HctTmp = 0.25
##############################
######
assign_local_Kw_Gmax = 1  # TODO: '0'=nonzero Kw and Gmax for ArtVen!!!
######
if benchPermeation == 1 or benchmarkAdvDiff == 1:
    dtAdjustedAutomatically = 0
    useCTCFCB = 0
    Ghypertrophy = 0.0
    ratioVtVb = 1.0
######
numCycle = 1  # TODO: set to '0' to use 'diffTimeNeeded' for 'num_steps'; Otherwise, use 'numCycle'!
######
##############################

##############################
gridName = "mesh"
with XDMFFile(MPI.COMM_WORLD, XDMFmeahFileName, "r") as xdmfTmp:
    domain = xdmfTmp.read_mesh(name=gridName)
    # Read tags:
    try:
        vertexTags = xdmfTmp.read_meshtags(domain, name="mesh_tags")
        cellTags = xdmfTmp.read_meshtags(domain, name="Cell tags")
        if MPI.COMM_WORLD.rank == 0:
            print(f"Mesh tags are read successfully on MPI_rank {rankMPI}.")
    except RuntimeError as e:
        if MPI.COMM_WORLD.rank == 0:
            print(f"Error reading mesh tags for cell: {e}")
        # vertexTags = None
        exit(0)


cell_indices = cellTags.indices
cell_values = cellTags.values
num_cells = len(cell_indices)

# Print out the indices and their corresponding values
vertex_indices = vertexTags.indices
vertex_values = vertexTags.values
for index, value in zip(vertex_indices, vertex_values):
    if value > 0 and MPI.COMM_WORLD.rank == 0:
        print(f"--> Vertex Index: {index}, Tag Value: {value}")

num_vertices = len(vertex_indices)
midVertexID = int(num_vertices * 0.5)

if MPI.COMM_WORLD.rank == 0:
    print(f"--> num_vertices: {num_vertices}")
    print(f"--> num_cells: {num_cells}")

# Facet (vertex in my case) dimension
fdim = domain.topology.dim - 1  # BC applied on vertices!

# Define tag values for inlets and outlets
inlet_tag_value = 1  # TODO
# outlet_tag_value = 2    # TODO
outlet_tag_value = [2]  # TODO: [2, 4]

inlet_facets_tmp = vertexTags.indices[vertexTags.values == inlet_tag_value]
# outlet_facets_tmp = vertexTags.indices[vertexTags.values == outlet_tag_value]

outlet_facets_list = []
for tagValue in outlet_tag_value:
    outlet_facets_list.extend(vertexTags.indices[vertexTags.values == tagValue])
# Convert outlet_facets to a numpy array
outlet_facets_tmp = np.array(outlet_facets_list, dtype=np.int32)

print(f"--> Inlet Vertex ID: {inlet_facets_tmp}")
print(f"--> Outlet Vertex ID: {outlet_facets_tmp}")
# exit(0)

#########################################################
# Initialize XDMF files for writing results:
# NOTE: No need to guard the file writing operation with an if MPI.COMM_WORLD.rank == 0: condition.
# DOLFINx and the XDMF library handle the parallel I/O internally and correctly.

xdmf_blood = XDMFFile(domain.comm, os.path.join(folder_name, 'results_blood.xdmf'), "w")
xdmf_blood.write_mesh(domain)  # Write the mesh to the XDMF file

xdmf_Qnode = XDMFFile(domain.comm, os.path.join(folder_name, 'results_Qnode.xdmf'), "w")
xdmf_Qnode.write_mesh(domain)
xdmf_Rnode = XDMFFile(domain.comm, os.path.join(folder_name, 'results_Rnode.xdmf'), "w")
xdmf_Rnode.write_mesh(domain)

xdmf_tissue = XDMFFile(domain.comm, os.path.join(folder_name, 'results_tissue.xdmf'), "w")
xdmf_tissue.write_mesh(domain)

#########################################################
# Get coordinates of the domain
x = SpatialCoordinate(domain)

#########################################################
# Define function space, trial and test functions in parallel:
element = FiniteElement("CG", domain.ufl_cell(), degree=1)
element1 = FiniteElement("CG", domain.ufl_cell(), degree=1)
mixedElem = MixedElement([element, element1])
V = FunctionSpace(domain, mixedElem)

V0, V0toV = V.sub(0).collapse()
V1, V1toV = V.sub(1).collapse()

# Define trial, test and solution functions:
dudut = TrialFunction(V)
du, dut = split(dudut)
vvt = TestFunction(V)
v, vt = split(vvt)
uut = Function(V)
u, ut = split(uut)

ununt = Function(V)
un, unt = split(ununt)

V_dg = FunctionSpace(domain, ("DG", 0))
V_cg = FunctionSpace(domain, ("CG", 1))

#########################################################
### COMPUTE DIRECTIONAL VECTORS
direction_vector_cells = compute_directional_vectors_cells(domain)
dirVector_DG = cellDirVec_DG(domain, direction_vector_cells)

# Initialize solver and A outside of the for-loop
my_petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
target_vectorCGSpace = VectorFunctionSpace(domain, ("CG", 1))
solver_linear_dirVec, A_linear_dirVec = create_solver(target_vectorCGSpace, my_petsc_options)
dirVector = project_function(target_vectorCGSpace, dirVector_DG, solver_linear_dirVec, A_linear_dirVec)

if MPI.COMM_WORLD.Get_size() > 1:
    MPI.COMM_WORLD.barrier()  # TODO: NECESSARY?
    dirVector.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  
    MPI.COMM_WORLD.barrier()  


#########################################################
### Create CG function space for scalar variables:
solver_linear_scalar_Q, A_linear_scalar_Q = create_solver(V_cg, my_petsc_options)

solver_linear_scalar_R, A_linear_scalar_R = create_solver(V_cg, my_petsc_options)

if MPI.COMM_WORLD.Get_size() > 1:
    MPI.COMM_WORLD.barrier()  # TODO: NECESSARY?

# Initialize variables
BCLmat = None
QvesselMatlab = None
RvesselMatlab = None
Vvessel_tot = None

if rankMPI == 0:
    # Load data only on the root process
    matdata = loadmat(str(matDataPath) + 'hemo_cleaned_with_control.mat')
    BCLmat_list = matdata['BCL']
    BCLmat = int(BCLmat_list[0][0])  # use 'int' otherwise '[-BCLmat:]' give null!!!
    matlabQO = 'Qo_log'  # 'Qvessel' or 'Qvessel_constQ'
    matlabQi = 'Qi_log'
    Qo_mat = matdata[matlabQO] * 1000.0  #[mL/s] --> [mm^3/s]
    Qi_mat = matdata[matlabQi] * 1000.0  #[mL/s] --> [mm^3/s]
    Qvessel_mat = 0.5*(Qo_mat + Qi_mat)  # [mm^3/s]
    Qvessel_array_all = np.asfarray(Qvessel_mat, float)
    Qvessel_array = Qvessel_array_all[-BCLmat:]  # Take only [-BCL:END][vID]

    Rvessel_mat = matdata['Rat_log']  # [mm]
    Rvessel_array_all = np.asfarray(Rvessel_mat, float)
    Rvessel_array = Rvessel_array_all[-BCLmat:]
    print(f"--> len(Qvessel_array_all): {len(Qvessel_array_all)}")
    print(f"--> len(Qvessel_array): {len(Qvessel_array)}")
else:
    # Ensure other ranks wait for root to load the data
    pass

if rankMPI == 0:
    nTmp = 0
    QvesselMatlab = Qvessel_array[nTmp]
    RvesselMatlab = Rvessel_array[nTmp]
    print(f"--> Vvessel_tot: {Vvessel_tot}")  # ; exit(0)
else:
    pass  # Non-root processes will wait for the broadcast

# Broadcast the data from root to all other processes
if MPI.COMM_WORLD.Get_size() > 1:
    MPI.COMM_WORLD.barrier()  # TODO: NOT NECESSARY??
    BCLmat = commMPI.bcast(BCLmat, root=0)
    QvesselMatlab = commMPI.bcast(QvesselMatlab, root=0)
    RvesselMatlab = commMPI.bcast(RvesselMatlab, root=0)

# Create DG function space (DG0)
Qcell_function = Function(V_dg)
Rcell_function = Function(V_dg)

# Get the local range for the process
local_range = V_dg.dofmap.index_map.local_range
local_start, local_end = local_range  # Unpack the local range
ghost_indices = V_dg.dofmap.index_map.ghosts

# Generate QvesselFEM for local cells
QvesselFEM = [QvesselMatlab[int(vID - 1)] for cID, vID in zip(cell_indices, cell_values)]
RvesselFEM = [RvesselMatlab[int(vID - 1)] for cID, vID in zip(cell_indices, cell_values)]

# Assign values to the DG function for local cells
with Qcell_function.vector.localForm() as loc:
    for cell in range(len(QvesselFEM)):
        dof = V_dg.dofmap.cell_dofs(cell)[0]  # One dof per cell for DG0
        loc.setValue(dof, QvesselFEM[cell])

with Rcell_function.vector.localForm() as loc:
    for cell in range(len(RvesselFEM)):
        dof = V_dg.dofmap.cell_dofs(cell)[0]  # One dof per cell for DG0
        loc.setValue(dof, RvesselFEM[cell])

# Update ghost cells correctly across processes
if MPI.COMM_WORLD.Get_size() > 1:
    Qcell_function.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    Rcell_function.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    MPI.COMM_WORLD.barrier()  # MUST ensure all processes are synchronized here!

# Project the DG function to CG function
# Qnode_CG = fem.Function(V_cg)
Qnode_CG = project_function(V_cg, Qcell_function, solver_linear_scalar_Q, A_linear_scalar_Q)
Rnode_CG = project_function(V_cg, Rcell_function, solver_linear_scalar_R, A_linear_scalar_R)

# Ensure synchronization after projection
if MPI.COMM_WORLD.Get_size() > 1:
    MPI.COMM_WORLD.barrier()  # Ensure all processes here!
    Qnode_CG.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # TODO: NECESSARY?
    Rnode_CG.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # TODO: NECESSARY?
    MPI.COMM_WORLD.barrier()  # TODO: NECESSARY?


if rankMPI == 0:
    print(f'--> BCLmat: {BCLmat}')
    Qcg_all_tmp = Qnode_CG.vector.getArray(readonly=True)
    Rcg_all_tmp = Rnode_CG.vector.getArray(readonly=True)
    print(f'Qnode_CG[0:6] (mm^3/s): {Qcg_all_tmp[:6]}')
    print(f'QvesselFEM[0:6]: {QvesselFEM[:6]}')
    print(f'Rnode_CG[0:6] (mm): {Rcg_all_tmp[:6]}')
    print(f'RvesselFEM[0:6]: {RvesselFEM[:6]}')

#########################################################
# Model parameters
dL_cell = CellDiameter(domain)  # [mm]
solver_linear_scalar_dL, A_linear_scalar_dL = create_solver(V_cg, my_petsc_options)
dL = project_function(V_cg, dL_cell, solver_linear_scalar_dL, A_linear_scalar_dL)
#########################################################
dL_new = dL  # TODO: So no need to replace 'dL_new' by 'dL' when not using 'dL_new'.
#########################################################
#########################################################
if rankMPI == 0:
    print(f'--> CHECK: dL_new = {dL_new.vector.getArray(readonly=True)}')

####################################
###### Better to compute Area and Volume from cells and then map to vertices!!!
Asurface_cell = 2.0 * np.pi * Rcell_function * dL_cell  # Surface area!!!
AcrossSect_cell = np.pi * Rcell_function ** 2  # Cross-sectional area!!!
Vb_cell = AcrossSect_cell * dL_cell  # Volume of blood or plasma
# Mapping uut from vertices to cells:
solver_linear_scalar_dg2cg_, A_linear_scalar_dg2cg_ = create_solver(V_cg, my_petsc_options)
Asurface = project_function(V_cg, Asurface_cell, solver_linear_scalar_dg2cg_, A_linear_scalar_dg2cg_)
AcrossSect = project_function(V_cg, AcrossSect_cell, solver_linear_scalar_dg2cg_, A_linear_scalar_dg2cg_)
Vb = project_function(V_cg, Vb_cell, solver_linear_scalar_dg2cg_, A_linear_scalar_dg2cg_)

#########################################################
difD = difD_value

if useCTCFCB == 1:
    difDHb = difD
elif useCTCFCB == 2:
    difDHb = difD / 65.0
else:
    difDHb = 0.0  # TODO

#########################################################
if findMaxPeMaxCFL == 1:
    Qnode_DG_max = np.max(Qvessel_array[:])
    Rnode_DG_max = np.min(Rvessel_array[:])
    advUmax = Qnode_DG_max / (np.pi * Rnode_DG_max ** 2)
    dLmin = np.min(dL.vector.getArray(readonly=True))
    PeMax = advUmax * dLmin / (2 * difD)
    CFLmax = advUmax * dtTmp / dLmin
    if MPI.COMM_WORLD.rank == 0:
        print(f'--> Qnode_DG_max = {Qnode_DG_max}')
elif findMaxPeMaxCFL == 2:
    # Calculate velocity
    if isSingleVessel == 1:
        U_array_allTimes = Qvessel_array / (np.pi * Rvessel_array ** 2)
        U_array = U_array_allTimes[0]  # TODO: This is for U_cells_t0!
    else:
        U_array = Qnode_CG.vector.getArray(readonly=True) / (np.pi * (Rnode_CG.vector.getArray(readonly=True)) ** 2)

    dL_array = dL.vector.getArray(readonly=True)
    
    Pe_array = (U_array * dL_array) / (2 * difD)
    CFL_array = U_array * dtTmp / dL_array
    
    # Find the absolute maximum value in the Pe_array
    PeMax = np.max(np.abs(Pe_array))
    CFLmax = np.max(np.abs(CFL_array))
else:
    print("--> ERROR: TRY 'findMaxPeMaxCFL = 1' or REDUCE 'CFLreduceFactor' TO 10... !!!")
    exit(0)

MPI.COMM_WORLD.barrier()  # TODO: NOT NECESSARY??
if MPI.COMM_WORLD.rank == 0:
    print('--> Max Peclet number (AdvectionRate/DiffusionRate) = ', PeMax)
    print('--> Max CFL (TravelDistance/GridSize) = ', CFLmax)
    #exit(0)
    print(
        f'CHECK: CFLmax<=0.1 worked well! --> dt_old={dtTmp}, dt_new={dtTmp / (CFLmax * CFLreduceFactor)} for CFLmax={1.0 / CFLreduceFactor}')
#########################################################
dt = dtTmp
if dtAdjustedAutomatically == 1:
    dt = dtTmp / (CFLmax * CFLreduceFactor)
    dt_new = dtTmp / (CFLmax * CFLreduceFactor)

if isNonPulsatileCase == 0 and dt > dtTmp:
    dt = dtTmp  # Keep 'dt=dt_MatLab' for pulsatile cases, especially if dt>0.001!
    dt_new = dtTmp

dtValue = dt
BCLfem = int(0.8 / dt)
#diffTimeNeeded = int(pathLenTmp * pathLenTmp / difD)  # TODO: 'L^2/difD' [seconds]

if numCycle > 0:
    num_steps = int(numCycle * BCLfem)  # Before 'Constant(domain, dt)!!! 'math.ceil(numCycle / dt)'
else:
    num_steps = int(diffTimeNeeded / dt)
    numCycleTemp = int(num_steps / BCLfem) 
    num_steps = int(numCycleTemp * BCLfem)
 
######
uInValue = constPO2inlet * pO2C  # [mmHg] --> [mol/mm^3]; 100
uOutValue = 20.0 * pO2C  # [mmHg] --> [mol/mm^3]; 20.0; CSpO2=~20mmHg

if benchmarkAdvDiff == 1:
    uInValue = 0.0  # This is not OK if Gmax!=0.
    uOutValue = 0.0
    initMethod = 'initGaussianHill'

init_value_CFb = uInValue * 0.8
init_value_CFt = uOutValue  # myoPO2=10-30mmHg

if benchPermeation == 1:
    uInValue = 95.0 * pO2C  # [mmHg] --> [mol/mm^3];
    uOutValue = uInValue
    init_value_CFb = uOutValue
    init_value_CFt = 0.0

if rankMPI == 0:
    print(f'--> Concentration [mol/mm^3]: inlet={uInValue}; outlet={uOutValue}')
    print(f'pO2 [mmHg]: inlet={uInValue / pO2C}; outlet={uOutValue / pO2C}')

######
advU = Qnode_CG / AcrossSect  # Flow velocity [mm/s]=[mm^3/s]/[mm^2]!
CHb = Constant(domain, 5.3 * 1e-9)  # [1mM = 1e-9mol/mm^3]
######
kWtmp = kWratioTmp * 35.0 * 0.001  # [DB2001] [um/s] --> [mm/s]  [Ref: 0.1-6 or 10 or 35 or 50um/s!]
######
if benchmarkAdvDiff == 1:
    kWtmp = 0.0

if assign_local_Kw_Gmax == 1:
    kW = assign_local_property_vertexBased(domain, kWtmp, V0)
    kW.name = f"kW"
    if rankMPI == 0:
        print(f'--> CHECK: kW = {kW.vector.getArray(readonly=True)}')
else:
    kW = Constant(domain, PETSc.ScalarType(kWtmp))

if benchPermeation == 1:
    advU = Constant(domain, 0.0)
    difD = 0.0

if rankMPI == 0:
    print(f'--> CHECK: advU = {advU}')

#########################################################
# Define Péclet and Courant numbers on all processes
if difD > 0.0:
    Pe = advU * dL / (2 * difD)
else:
    Pe = 'inf'
CFL = advU * dt / dL

MPI.COMM_WORLD.barrier()  # TODO: NOT NECESSARY??

if MPI.COMM_WORLD.rank == 0:
    local_dL_values = dL.vector.getArray(readonly=True)
    first_two_dL_values = local_dL_values[:2]
    print(f'--> dL={first_two_dL_values}; dt={dt}; num_steps={num_steps}')

#########################################################
dt = Constant(domain, dt)
difD = Constant(domain, difD)
difDHb = Constant(domain, difDHb)

######
Vtis = Vb * ratioVtVb
######
AkVt = Asurface * kW / Vtis
AkVb = Asurface * kW / Vb

if MPI.COMM_WORLD.Get_size() > 1:
    MPI.COMM_WORLD.barrier()  # TODO: NECESSARY?

#########################################################
# Define boundary conditions:
#########################################################
# Locate DOFs for inlets and outlets
dofs_inlet_list = locate_dofs_topological(V.sub(0), fdim, np.array([inlet_facets_tmp], dtype=np.int32))

dofs_outlet_list = locate_dofs_topological(V.sub(0), fdim, np.array([outlet_facets_tmp], dtype=np.int32))

# Convert each element to a NumPy array and get its string representation
dofs_inlet_list = [(np.array([item], dtype=np.int32)) for item in dofs_inlet_list]
dofs_outlet_list = [(np.array([item], dtype=np.int32)) for item in dofs_outlet_list]

# Define the boundary conditions
bc_inlets_list = []
bc_outlets_list = []
for dofs_i in dofs_inlet_list:
    bcInlet_ = dirichletbc(value=PETSc.ScalarType(uInValue), dofs=dofs_i, V=V.sub(0))
    bc_inlets_list.append(bcInlet_)

for dofs_o in dofs_outlet_list:
    bcOutlet_ = dirichletbc(value=PETSc.ScalarType(uOutValue), dofs=dofs_o, V=V.sub(0))
    bc_outlets_list.append(bcOutlet_)

# Combine boundary conditions into a list
if benchPermeation == 1:
    bcsb_ = []  # TODO: Free all openings
else:
    bcsb_ = [bc_inlets_list]  # TODO: Free outlets

# Flatten the list so that each element becomes a separate sublist
bcsb = [item for sublist in bcsb_ for item in sublist]

#########################################################
# Define variational problem (F, J) in parallel
#########################################################
######### Blood Domain - Hemaglobin-bonded O2:
Hct = Constant(domain, HctTmp)
######
W = as_matrix([[7.0 / 24.0, -1.0 / 24.0], [13.0 / 24.0, 5.0 / 24.0]])
W_inv = inv(W)
ww = as_vector([0.5, 0.5])
vv = as_vector([v, v])
Pw = W * advU * dot(grad(v), dirVector)
######
CBn = variable(4.0 * CHb * Hct * SHb(domain, un, pO2C))
CB = variable(4.0 * CHb * Hct * SHb(domain, u, pO2C))

######

################## COMBINED CF and CB into a single Fb:
CTn = CBn + un
CT = CB + u
CTn = variable(CTn)
CT = variable(CT)
######### Blood Domain - Free O2:
deltaCF = as_vector([((u + un) * 0.5 - un), (u - (u + un) * 0.5)])
deltaCF = variable(deltaCF)

if useCTCFCB > 0:
    deltaU = as_vector([((CT + CTn) * 0.5 - CTn), (CT - (CT + CTn) * 0.5)])
else:
    deltaU = as_vector([((u + un) * 0.5 - un), (u - (u + un) * 0.5)])
deltaU = variable(deltaU)

WU = W * deltaU
WU = variable(WU)
WCF = W * deltaCF
# WCF = W * as_vector([(u - un), (u - un)])   # TODO: W*(u-un) ?
WCF = variable(WCF)

#########
deltaSb = as_vector([(AkVb * (ut + unt) * 0.5 - AkVb * unt), (AkVb * ut - AkVb * (ut + unt) * 0.5)])
deltaSb = variable(deltaSb)  # TODO: 'S=AkVb*CFt'


# L-Operator:
def weakL(v, CF, CT):
    if useCTCFCB == 1:
        return v * advU * (dot(grad(CT), dirVector)) \
                + difD * inner(grad(CF), grad(v)) + difDHb * inner(grad(CB), grad(v)) + v * AkVb * CF
    elif useCTCFCB == 2:
        return v * advU * (dot(grad(CT), dirVector)) \
                + difD * inner(grad(CT), grad(v)) + v * AkVb * CF
    else:
        return v * advU * (dot(grad(CF), dirVector)) \
                + difD * inner(grad(CF), grad(v)) + v * AkVb * CF


# Residual term for stabilization:
def funR(deltaCT, deltaCF, deltaS, CFn, CTn, CFtn):
    if useCTCFCB == 1:
        gradDeltaCT = as_vector([dot(grad(deltaCT[0]), dirVector),
                                    dot(grad(deltaCT[1]), dirVector)])
        return deltaCT / dt + W * advU * gradDeltaCT + W * AkVb * deltaCF - ww * (
                AkVb * CFtn - advU * (dot(grad(CTn), dirVector)) - AkVb * CFn) - W * deltaS
    elif useCTCFCB == 2:
        gradDeltaCT = as_vector([dot(grad(deltaCT[0]), dirVector),
                                    dot(grad(deltaCT[1]), dirVector)])
        return deltaCT / dt + W * advU * gradDeltaCT + W * AkVb * deltaCF - ww * (
                AkVb * CFtn - advU * (dot(grad(CTn), dirVector)) - AkVb * CFn) - W * deltaS
    else:
        gradDeltaCF = as_vector([dot(grad(deltaCF[0]), dirVector),
                                    dot(grad(deltaCF[1]), dirVector)])
        return deltaCF / dt + W * advU * gradDeltaCF + W * AkVb * deltaCF - ww * (
                AkVb * CFtn - advU * (dot(grad(CFn), dirVector)) - AkVb * CFn) - W * deltaS


##################
# Use SUPG + R22:
if steadySUPG == 1:  # useQatSpecificTime == 1 or constQFlag == 1:
    Pe_temp = advU * dL / (2 * (difD + difDHb))
    tau = (dL / (2.0 * advU)) * (1.0 / ufl.tanh(Pe_temp) - 1.0 / Pe_temp) * W_inv
else:
    tauC = 2.0 * advU / dL + 4.0 * difD / dL ** 2 + AkVb
    if useCTCFCB == 1:
        tauC = 2.0 * advU / dL + 4.0 * (difD + difDHb) / dL ** 2 + AkVb
    tau = inv(W_inv / dt + tauC * Identity(2)).T * W_inv

tauPw = tau * Pw
tauPwR = tauPw * funR(deltaU, deltaCF, deltaSb, un, CTn, unt)
tauPwR = variable(tauPwR)  # TODO

# TODO: Update 'PeMax' inside the for-loop!!!
if findMaxPeMaxCFL == 2:
    PeMax = np.max(np.abs((U_array * dL_array) / (2 * difD_value)))
    print(f"At Step 0: PeMax = {PeMax}")
# TODO: Use SUPG only if 'PeMax > PeCritical':
if PeMax > PeCritical:
    Fb = dot(deltaU / dt, vv) * dx \
            + (weakL(v, WCF[0], WU[0]) + weakL(v, WCF[1], WU[1])) * dx \
            + (ww[0] * weakL(v, un, CTn) + ww[1] * weakL(v, un, CTn)) * dx \
            - (ww[0] * AkVb * unt * v + ww[1] * AkVb * unt * v) * dx \
            - dot(W * deltaSb, vv) * dx \
            + (tauPwR[0] + tauPwR[1]) * dx
else:
    Fb = dot(deltaU / dt, vv) * dx \
            + (weakL(v, WCF[0], WU[0]) + weakL(v, WCF[1], WU[1])) * dx \
            + (ww[0] * weakL(v, un, CTn) + ww[1] * weakL(v, un, CTn)) * dx \
            - (ww[0] * AkVb * unt * v + ww[1] * AkVb * unt * v) * dx \
            - dot(W * deltaSb, vv) * dx

#########################################################
###### Tissue Domain:
maxGvalue = 70.0 * 1E-12   # [mol/(mm^3 s)]-->[uM/s]; 1uM=1e-12mol/mm^3
if initMethod == 'initDeltaPulse':
    maxGvalue = 0.0

if assign_local_Kw_Gmax == 1:
    maxG = assign_local_property_vertexBased(domain, maxGvalue, V0)  
else:
    maxG = Constant(domain, PETSc.ScalarType(maxGvalue))

if rankMPI == 0 and assign_local_Kw_Gmax == 1:
    print(f'--> CHECK: maxG = {maxG.vector.getArray(readonly=True)}')

km = Constant(domain, 1e-7 * 1e-6) 

#Dmb = Constant(domain, 0.0)
Dmb = Constant(domain, 2.2e-7 * 100)  # Diffusion coefficient: 2.2×10−7cm2/s (cm^2/s=100mm^2/s).
CMb = Constant(domain, 1E-4 * 1E-6)  # 1E-4M (1M=1E-6mol/mm^3)
C50 = Constant(domain, 2.5 * mmHg_to_mmGs)  # 2.5Torr [1Torr = 1mmHg] [1 mmHg = 133.322 g/(mm·s²)]

if useR22ForTissue == 1:
    ####### USING R22:
    deltaCFt = as_vector([((ut + unt) * 0.5 - unt), (ut - (ut + unt) * 0.5)])
    deltaCFt = variable(deltaCFt)
    deltaSt = as_vector([(AkVt * (u + un) * 0.5 - AkVt * un), (AkVt * u - AkVt * (u + un) * 0.5)])
    deltaSt = variable(deltaSt)  # TODO: 'S=AkVt*CF'
    vtvt = as_vector([vt, vt])

    WUt = W * deltaCFt
    WUt = variable(WUt)

    ### L-Operator:
    def weakLtissue(vt, CFt):
        if wTissueDiffusion == 0:
            return AkVt * CFt * vt + (maxG * CFt / (CFt + km)) * vt
        else:
            return difD * inner(grad(CFt), grad(vt)) + Dmb * CMb * inner(grad(CFt / (CFt + C50)), grad(vt)) \
                   + AkVt * CFt * vt + (maxG * CFt / (CFt + km)) * vt
    
    ### Generic for BOTH wo/w diffusion in tissue:
    Ft = dot(deltaCFt / dt, vtvt) * dx \
         + (weakLtissue(vt, WUt[0]) + weakLtissue(vt, WUt[1])) * dx \
         - dot(W * deltaSt, vtvt) * dx \
         - (ww[0] * AkVt * un * vt + ww[1] * AkVt * un * vt) * dx \
         + (ww[0] * weakLtissue(vt, unt) + ww[1] * weakLtissue(vt, unt)) * dx
else:
    ####### Without USING R22:
    if wTissueDiffusion == 0:
        Ft = ((ut - unt) / dt) * vt * dx - AkVt * (u - ut) * vt * dx + (maxG * ut / (ut + km)) * vt * dx
    else:
        Ft = ((ut - unt) / dt) * vt * dx - AkVt * (u - ut) * vt * dx + (maxG * ut / (ut + km)) * vt * dx + \
             difD * inner(grad(ut), grad(vt)) * dx + \
             Dmb * CMb * inner(grad(ut / (ut + C50)), grad(vt)) * dx

#########################################################
#########################################################
# Define the combined form for F and Jac:
#########
if benchPermeation == 1 and useR22ForTissue == 0:
    Fb = ((u - un) / dt) * v * dx + AkVb * (u - ut) * v * dx
elif benchPermeation == 1 and useR22ForTissue == 1:
    ####### USING R22:
    Fb = dot(deltaCF / dt, vv) * dx \
         + dot(W * (AkVb * deltaCF - deltaSb), vv) * dx \
         - dot(ww * (AkVb * unt - AkVb * un), vv) * dx

#########
F = Fb + Ft
Jac = derivative(F, uut, dudut)

#########################################################
# Initialize function for initial conditions

# Blood Domain:
if initMethod == 'initConstant':
    init_cond = Function(V0)
    with init_cond.vector.localForm() as loc:
        loc.set(PETSc.ScalarType(init_value_CFb))
    ununt.sub(0).interpolate(init_cond)
elif initMethod == 'initGaussianHill':
    myGaussianHill = GaussianHill()
    ununt.sub(0).interpolate(myGaussianHill.eval)
elif initMethod == 'initDeltaPulse':
    myDeltaPulse = DeltaPulse()
    ununt.sub(0).interpolate(myDeltaPulse.eval)
else:
    if MPI.COMM_WORLD.rank == 0:
        print(f'--> ERROR: initMethod NOT DEFINED! TRY initGaussianHill OR initDeltaPulse OR initConstant ...')
    exit(0)

# Initialize for Tissue Domain
init_cond_tissue = Function(V1)
with init_cond_tissue.vector.localForm() as loc:
    loc.set(init_value_CFt)
# unt.interpolate(init_cond_tissue)
ununt.sub(1).interpolate(init_cond_tissue)

if MPI.COMM_WORLD.Get_size() > 1:
    MPI.COMM_WORLD.barrier()  # TODO: necessary?
    ununt.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    MPI.COMM_WORLD.barrier()

# Assign un to u & unt to ut
uut.sub(0).vector.array[:] = ununt.sub(0).vector.array
uut.sub(1).vector.array[:] = ununt.sub(1).vector.array

# Synchronize solution vectors across processes:  TODO: necessary?
if MPI.COMM_WORLD.Get_size() > 1:
    MPI.COMM_WORLD.barrier()  # TODO: NECESSARY?
    uut.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    MPI.COMM_WORLD.barrier()

#########################################################

#########################################################
problem = NonlinearProblem(F, uut, bcs=bcsb, J=Jac)
solver = NewtonSolver(domain.comm, problem)
solver.convergence_criterion = convergCriteria
solver.rtol = 1e-8
solver.report = True
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = kspType_
opts[f"{option_prefix}pc_type"] = pcType_
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
opts[f"{option_prefix}ksp_max_it"] = maxIter_ksp
ksp.setFromOptions()

##################################################################
if MPI.COMM_WORLD.rank == 0:  # Only the root process creates the folder
    create_folder_if_not_exists('stateBackup')

# Initialize or load simulation state
step_start = 0
if MPI.COMM_WORLD.Get_size() == 1:
    try:
        step_start = load_state(uut, ununt)
        print(f"Resuming from step {step_start}.")
    except FileNotFoundError:
        step_start = 0
        print("Starting new simulation.")

##################################################################
(u, ut) = uut.split()

######
if benchPermeation == 1:
    CFbTmp = uut.vector.array[V0toV]
    CFtTmp = uut.vector.array[V1toV]
    csvFile_writer.writerow([0, CFbTmp[midVertexID], CFtTmp[midVertexID]])
if benchmarkAdvDiff == 1:
    CFbTmp = uut.vector.array[V0toV]
    csvFile_writer.writerow([0, CFbTmp])


##################################################################
# Mapping uut from vertices to cells:
solver_linear_scalar_cg2dg_, A_linear_scalar_cg2dg_ = create_solver(V_dg, my_petsc_options)

# Time-stepping loop:
t = dt * step_start  # TODO: =0?
if benchmarkAdvDiff == 1 or benchPermeation == 1:
    num_steps = 1000  # TODO: FOR DEBUGGING...
# num_steps = 24  #TODO
#num_steps = int(BCLfem * numO2cycleIn1CardiacCycle *50)  #TODO

if numSteps_test > 0:
    num_steps = numSteps_test
       
# Adjust the simulation loop to start from step_start    
# for n in range(num_steps):
for n in range(step_start, num_steps):
    t += dt
    if n % 100 == 0 and rankMPI == 0:
        print(f'--> Current Step:{n}; Total Steps:{num_steps}; BCLfem:{BCLfem}')

    ### Update variables:
    # Determine the index of QR_matlab:
    if abs(BCLmat - BCLfem) < 0.5:
        nTmp = n % BCLmat  # Remainder
    else:
        nTmpFEM = n % BCLfem
        nTmp = int(nTmpFEM * BCLmat / BCLfem)
    # Read QR_matlab:
    if rankMPI == 0:
        QvesselMatlab = Qvessel_array[nTmp]
        RvesselMatlab = Rvessel_array[nTmp]
    else:
        pass  # Non-root processes will wait for the broadcast

    # Broadcast the data from root to all other processes
    if MPI.COMM_WORLD.Get_size() > 1:
        # MPI.COMM_WORLD.barrier()  #TODO: NOT NECESSARY??
        QvesselMatlab = commMPI.bcast(QvesselMatlab, root=0)
        RvesselMatlab = commMPI.bcast(RvesselMatlab, root=0)

    # Generate QvesselFEM & RvesselFEM for local cells
    QvesselFEM = [QvesselMatlab[int(vID - 1)] for cID, vID in zip(cell_indices, cell_values)]
    RvesselFEM = [RvesselMatlab[int(vID - 1)] for cID, vID in zip(cell_indices, cell_values)]

    # Assign values to the DG function for local cells
    # CASE-1: QvesselFEM & RvesselFEM are already LOCAL:
    with Qcell_function.vector.localForm() as loc:
        for cell in range(len(QvesselFEM)):
            dof = V_dg.dofmap.cell_dofs(cell)[0]  # One dof per cell for DG0
            loc.setValue(dof, QvesselFEM[cell])

    with Rcell_function.vector.localForm() as loc:
        for cell in range(len(RvesselFEM)):
            dof = V_dg.dofmap.cell_dofs(cell)[0]  # One dof per cell for DG0
            loc.setValue(dof, RvesselFEM[cell])

            # Update ghost cells correctly across processes
    if MPI.COMM_WORLD.Get_size() > 1:
        Qcell_function.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        Rcell_function.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        MPI.COMM_WORLD.barrier()  # Must ensure all processes are synchronized here!

    # Project the DG function to CG function
    Qnode_CG = project_function(V_cg, Qcell_function, solver_linear_scalar_Q, A_linear_scalar_Q)
    Rnode_CG = project_function(V_cg, Rcell_function, solver_linear_scalar_R, A_linear_scalar_R)

    # Ensure synchronization after projection
    if MPI.COMM_WORLD.Get_size() > 1:
        MPI.COMM_WORLD.barrier()
        Qnode_CG.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # TODO: NECESSARY?
        Rnode_CG.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # TODO: NECESSARY?
        MPI.COMM_WORLD.barrier()  # TODO: NECESSARY?

    #########################################################
    # Calculate PeMax for the current time step
    if findMaxPeMaxCFL == 2:
        if isSingleVessel == 1:
            U_array = U_array_allTimes[nTmp]  # TODO: This is for U_cells_tn!
        else:
            U_array = Qnode_CG.vector.getArray(readonly=True) / (np.pi * (Rnode_CG.vector.getArray(readonly=True)) ** 2)
        ### Update Fb based on PeMax: Use SUPG only if 'PeMax > PeCritical':
        PeMax = np.max(np.abs((U_array * dL_array) / (2 * difD_value)))
        if n <= step_start + BCLfem or n >= num_steps - BCLfem:
            print(f"At step {n}/{num_steps}:  PeMax = {PeMax}")

        if PeMax > PeCritical:
            Fb = dot(deltaU / dt, vv) * dx \
                    + (weakL(v, WCF[0], WU[0]) + weakL(v, WCF[1], WU[1])) * dx \
                    + (ww[0] * weakL(v, un, CTn) + ww[1] * weakL(v, un, CTn)) * dx \
                    - (ww[0] * AkVb * unt * v + ww[1] * AkVb * unt * v) * dx \
                    - dot(W * deltaSb, vv) * dx \
                    + (tauPwR[0] + tauPwR[1]) * dx
        else:
            Fb = dot(deltaU / dt, vv) * dx \
                    + (weakL(v, WCF[0], WU[0]) + weakL(v, WCF[1], WU[1])) * dx \
                    + (ww[0] * weakL(v, un, CTn) + ww[1] * weakL(v, un, CTn)) * dx \
                    - (ww[0] * AkVb * unt * v + ww[1] * AkVb * unt * v) * dx \
                    - dot(W * deltaSb, vv) * dx

    #########################################################
    #########################################################
    
    ### Solve the nonlinear problem
    num_its, converged = solver.solve(uut)
    # Print and check residual norm after each solve
    residual_norm = solver.krylov_solver.getResidualNorm()
    print(f"Step {n}: Residual norm = {residual_norm}")
    if np.isnan(residual_norm):
        print("Residual is NaN! Exiting simulation.")
        #sys.exit(1)
    elif residual_norm < 1e-8:
        print("Residual is sufficiently small. Converged.")

    # Synchronize solution vectors across processes:
    if MPI.COMM_WORLD.Get_size() > 1:
        MPI.COMM_WORLD.barrier()
        uut.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        MPI.COMM_WORLD.barrier()  # TODO: NECESSARY?

    (u, ut) = uut.split()

    # Ensure that values of CFt do not fall below zero
    ut_vals = ut.vector.array[:]  # Extract as NumPy array
    ut_vals[ut_vals < 0] = 0  # Ensure non-negativity
    ut.vector.array[:] = ut_vals  # Assign the modified values back

    # Another barrier to ensure all processes have updated their ghost values
    if MPI.COMM_WORLD.Get_size() > 1:
        MPI.COMM_WORLD.barrier()

    # Update un, unt:
    ununt.sub(0).vector.array[:] = uut.sub(0).vector.array
    ununt.sub(1).vector.array[:] = uut.sub(1).vector.array

    if MPI.COMM_WORLD.Get_size() > 1:
        MPI.COMM_WORLD.barrier()  # TODO: NECESSARY?
        ununt.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        MPI.COMM_WORLD.barrier()  # TODO: NECESSARY?

    (un, unt) = ununt.split()

    if MPI.COMM_WORLD.Get_size() > 1:
        MPI.COMM_WORLD.barrier()

    # Write data to XDMF files:
    if n % 100 == 0:
        u.name = f"CFb"  # '/pO2C' --> [mmHg]
        xdmf_blood.write_function(u, n + 1)
        ut.name = f"CFt"
        xdmf_tissue.write_function(ut, n + 1)
        dL_new.name = f"dLnode (mm)"
        if benchmarkAdvDiff != 1:  # and constQFlag != 1:
            Qnode_CG.name = f"Qnode (mm^3/s)"  # If '/1000' --> [mL/s]
            xdmf_Qnode.write_function(Qnode_CG, n + 1)
        if benchmarkAdvDiff != 1:  # and constRFlag != 1:
            Rnode_CG.name = f"Rnode (mm)"
            xdmf_Rnode.write_function(Rnode_CG, n + 1)

    if MPI.COMM_WORLD.Get_size() > 1:
        MPI.COMM_WORLD.barrier()

    # Checkpointing (save state) every 'BCLfem' steps
    if n % BCLfem == 0 and MPI.COMM_WORLD.Get_size() == 1:
        # save_state_mpi(MPI.COMM_WORLD, n, uut, ununt)
        save_state(n, uut, ununt)
        if MPI.COMM_WORLD.rank == 0:
            print(f"Checkpoint saved at step {n}.")


# Close XDMF files in a MPI-safe way:
xdmf_blood.close()
xdmf_tissue.close()
xdmf_Qnode.close()
xdmf_Rnode.close()


dirVector_DG.vector.destroy()

#######################################################################
# Mapping uut from vertices to cells:


#######################################################################
# Measure the elapsed time on all processes
runTime = perf_counter() - startTime

# Print the runtime only on the root process
if MPI.COMM_WORLD.rank == 0:
    print("--> The simulation took in total {} s".format(runTime))
    print(f'--> kWtmp:{kWtmp}； Vtis:{Vtis}, BCLfem={BCLfem}, dtNew={dtValue}')
    
# Printing from All Processes:
# PETSc.Sys.Print(f"Rank {PETSc.COMM_WORLD.rank}: The simulation took in total {runTime}s.")
