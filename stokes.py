#!/usr/bin/env python
from dolfin import *

# Test for PETSc or Epetra
#if not has_linear_algebra_backend("PETSc") and not has_linear_algebra_backend("Epetra"):
#    info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
#    exit()

#if not has_krylov_solver_preconditioner("amg"):
#    info("Sorry, this demo is only available when DOLFIN is compiled with AMG "
#	 "preconditioner, Hypre or ML.");
#    exit()

# Load mesh
#mesh = UnitCube(8, 8, 8)
mesh = UnitSquare(10,10)

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q

# Boundaries
def right(x, on_boundary): return x[0] > (1.0 - DOLFIN_EPS)
def left(x, on_boundary): return x[0] < DOLFIN_EPS
def top_bottom(x, on_boundary):
    return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS

# No-slip boundary condition for velocity
#noslip = Constant((0.0, 0.0, 0.0))
noslip = Constant((0.0, 0.0))
bc0 = DirichletBC(W.sub(0), noslip, top_bottom)

# Inflow boundary condition for velocity
#inflow = Expression(("-sin(x[1]*pi)", "0.0", "0.0"))
inflow = Expression(("-sin(x[1]*pi)", "0.0"))
bc1 = DirichletBC(W.sub(0), inflow, right)

# Boundary condition for pressure at outflow
zero = Constant(0)
bc2 = DirichletBC(W.sub(1), zero, left)

# Collect boundary conditions
bcs = [bc0, bc1] #, bc2]

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
#f = Constant((0.0, 0.0, 0.0))
f = Constant((0.0, 0.0))
a = inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx
L = inner(f, v)*dx

# Form for use in constructing preconditioner matrix
b = inner(grad(u), grad(v))*dx + p*q*dx

# Assemble system
A, bb = assemble_system(a, L, bcs)

# Assemble preconditioner system
P, btmp = assemble_system(b, L, bcs)

# Create Krylov solver and AMG preconditioner
#solver = KrylovSolver("tfqmr", "amg")

# Associate operator (A) and preconditioner matrix (P)
#solver.set_operators(A, P)

# Solve
U = Function(W)
#solver.solve(U.vector(), bb)
solve(a == L, U, bcs)

# Get sub-functions
u, p = U.split()

# Save solution in VTK format
ufile_pvd = File("velocity.pvd")
ufile_pvd << u
pfile_pvd = File("pressure.pvd")
pfile_pvd << p

# Plot solution
plot(u)
plot(p)
interactive()
