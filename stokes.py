#!/usr/bin/env python
import math
import numpy as np
from dolfin import *






def main():
	# Load mesh
	#mesh = UnitCube(8, 8, 8)
	errs_u = []
	errs_p = []
	for i in [ 2**i for i in range(2,7) ]:
		#print('Solving stokes with i={}'.format(i))
		err_u, err_p = solve_stokes(i)
		errs_u += [err_u]
		errs_p += [err_p]
	errs_u = np.array(errs_u)
	errs_p = np.array(errs_p)
	eoc_u = np.log(errs_u[1:]/errs_u[:-1])/np.log(0.5)
	eoc_p = np.log(errs_p[1:]/errs_p[:-1])/np.log(0.5)
	print('err_u: ', errs_u)
	print('err_p: ', errs_p)
	print('eoc_u: ', eoc_u)
	print('eoc_p: ', eoc_p)


def solve_stokes(n_unknowns):
	u_ex = Expression(("-t*sin(x[0]*t)*sin(x[1]*t)",
										 "-t*cos(x[0]*t)*cos(x[1]*t)"),t=1)
	p_ex = Expression(("exp(t*x[0])+exp(t*x[1])"),t=1)
	#p_ex = Expression(("5"),t=1)

	dtu0 = "-sin(t*x[0])*sin(t*x[1]) - t*x[0]*cos(t*x[0])*sin(t*x[1]) - t*x[1]*sin(t*x[0])*cos(t*x[1]) "
	dtu1 = "-cos(t*x[0])*cos(t*x[1]) + t*x[0]*sin(t*x[0])*cos(t*x[1]) + t*x[1]*cos(t*x[0])*sin(t*x[1]) "
	lap_u0 = "- 2*t*t*t/Reynolds*sin(t*x[0])*sin(t*x[1]) "
	lap_u1 = "- 2*t*t*t/Reynolds*cos(t*x[0])*cos(t*x[1]) "
	nonlin0 = " +t*t*t*sin(t*x[0])*cos(t*x[0]) "
	nonlin1 = " -t*t**sin(t*x[1])*cos(t*x[1]) "
	grad_p0 = " + t*exp(t*x[0]) "
	grad_p1 = " + t*exp(t*x[1]) "

	f = Expression((lap_u0+grad_p0, lap_u1+grad_p1),Reynolds=1,t=1)
	#f = Expression((lap_u0, lap_u1),Reynolds=1,t=1)


	mesh = UnitSquare(n_unknowns,n_unknowns)
	#print(mesh.num_cells())

	#avg = assemble(p_ex*dx, mesh=mesh)
	#p_ex = Expression(("exp(t*x[0])+exp(t*x[1]) - {}".format(avg)),t=1)
	#avg = assemble(p_ex*dx, mesh=mesh)
	#print('avg:', avg)

	# Define function spaces
	V = VectorFunctionSpace(mesh, "CG", 2)
	Q = FunctionSpace(mesh, "CG", 1)
	L = FunctionSpace(mesh, "R", 0)
	W = MixedFunctionSpace([V,Q,L])

	# Boundaries
	def right(x, on_boundary): return x[0] > (1.0 - DOLFIN_EPS)
	def left(x, on_boundary): return x[0] < DOLFIN_EPS
	def top_bottom(x, on_boundary):
			return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS

	# No-slip boundary condition for velocity
	#noslip = Constant((0.0, 0.0, 0.0))
	#noslip = Constant((0.0, 0.0))

	bc0 = DirichletBC(W.sub(0), u_ex, lambda x, bd: bd)

#	# Inflow boundary condition for velocity
#	#inflow = Expression(("-sin(x[1]*pi)", "0.0", "0.0"))
#	inflow = Expression(("-sin(x[1]*pi)", "0.0"))
#	bc1 = DirichletBC(W.sub(0), inflow, right)
#
#	# Boundary condition for pressure at outflow
#	zero = Constant(0)
#	bc2 = DirichletBC(W.sub(1), zero, left)
#
#	# Collect boundary conditions
#	bcs = [bc0, bc1] #, bc2]
	bcs = [bc0]

	# Define variational problem
	(u, p, lam) = TrialFunctions(W)
	(v, q, l) = TestFunctions(W)
	#f = Constant((0.0, 0.0, 0.0))
	#f = Constant((0.0, 0.0))
	a = inner(grad(u), grad(v))*dx - div(v)*p*dx - q*div(u)*dx + lam*q*dx + p*l*dx
	L = inner(f, v)*dx + p_ex*l*dx


	# Form for use in constructing preconditioner matrix
#	b = inner(grad(u), grad(v))*dx + p*q*dx

	# Assemble system
#	A, bb = assemble_system(a, L, bcs)

	# Assemble preconditioner system
#	P, btmp = assemble_system(b, L, bcs)

	# Create Krylov solver and AMG preconditioner
	#solver = KrylovSolver("tfqmr", "amg")

	# Associate operator (A) and preconditioner matrix (P)
	#solver.set_operators(A, P)

	# Solve
	U = Function(W)
	#solver.solve(U.vector(), bb)
	solve(a == L, U, bcs, solver_parameters = {"linear_solver": "petsc"})

	# Get sub-functions
	u, p, lam = U.split()

	#print(norm(lam))

	# Save solution in VTK format
	#ufile_pvd = File("velocity.pvd")
	#ufile_pvd << u
	pfile_pvd = File("pressure.pvd")
	pfile_pvd << p
	#errfile_pvd = File("error_u.pvd")
	#errfile_pvd << err
	return (errornorm(u_ex,u),errornorm(p_ex,p))

	# Plot solution
	#plot(u)
	#plot(p)
	#interactive()


if __name__ == '__main__':
	main()
