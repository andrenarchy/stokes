#!/usr/bin/env python
import math
import numpy as np
from dolfin import *

def main():
    # Load mesh
    #mesh = UnitCube(8, 8, 8)
    errs_u = []
    errs_p = []
    #for i in [ 2**i for i in range(2,7) ]:
        #print('Solving stokes with i={}'.format(i))
    err_u, err_p = solve_stokes(2**4)
    print('err_u:',err_u)
    print('err_p:',err_p)


#       errs_u += [err_u]
#       errs_p += [err_p]
#   errs_u = np.array(errs_u)
#   errs_p = np.array(errs_p)
#   eoc_u = np.log(errs_u[1:]/errs_u[:-1])/np.log(0.5)
#   eoc_p = np.log(errs_p[1:]/errs_p[:-1])/np.log(0.5)
#   print('err_u: ', errs_u)
#   print('err_p: ', errs_p)
#   print('eoc_u: ', eoc_u)
#   print('eoc_p: ', eoc_p)


def solve_stokes(n_unknowns):
    u_ex = Expression(("-t*sin(x[0]*t)*sin(x[1]*t)",
                                         "-t*cos(x[0]*t)*cos(x[1]*t)"),t=0)
    p_ex = Expression(("exp(t*x[0])+exp(t*x[1])"),t=0)

    dtu0 = "-sin(t*x[0])*sin(t*x[1]) - t*x[0]*cos(t*x[0])*sin(t*x[1]) - t*x[1]*sin(t*x[0])*cos(t*x[1]) "
    dtu1 = "-cos(t*x[0])*cos(t*x[1]) + t*x[0]*sin(t*x[0])*cos(t*x[1]) + t*x[1]*cos(t*x[0])*sin(t*x[1]) "
    lap_u0 = "- 2*t*t*t/Reynolds*sin(t*x[0])*sin(t*x[1]) "
    lap_u1 = "- 2*t*t*t/Reynolds*cos(t*x[0])*cos(t*x[1]) "
    nonlin0 = " +t*t*t*sin(t*x[0])*cos(t*x[0]) "
    nonlin1 = " -t*t**sin(t*x[1])*cos(t*x[1]) "
    grad_p0 = " + t*exp(t*x[0]) "
    grad_p1 = " + t*exp(t*x[1]) "

    f = Expression((lap_u0+grad_p0+dtu0, lap_u1+grad_p1+dtu1),Reynolds=1,t=0)

    mesh = UnitSquare(n_unknowns,n_unknowns)

    # Define function spaces
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    L = FunctionSpace(mesh, "R", 0)
    W = MixedFunctionSpace([V,Q,L])
    U = Function(W)

    t0 = 0
    tend = 1
    dt = .05

    # Initial value
    u_old = Constant((0,0))

    # Define variational problem
    (u, p, lam) = TrialFunctions(W)
    (v, q, l) = TestFunctions(W)
    a = inner(u,v)*dx + dt*(inner(grad(u), grad(v))*dx - div(v)*p*dx) - q*div(u)*dx + lam*q*dx + p*l*dx
    bc = DirichletBC(W.sub(0), u_ex, lambda x, bd: bd)


    ufile_pvd = File("velocity.pvd")
    pfile_pvd = File("pressure.pvd")
    errs_u = []
    errs_p = []
    for t in np.arange(t0,tend,dt):
        # Solve
        u_ex.t = t+dt
        p_ex.t = t+dt
        f.t = t+dt

        L = inner(u_old,v)*dx + dt*(inner(f, v)*dx) + p_ex*l*dx
        solve(a == L, U, bc, solver_parameters = {"linear_solver": "petsc"})
        # Get sub-functions
        u_new, p_new, lam_new = U.split()
        errs_u += [errornorm(u_ex,u_new)]
        errs_p += [errornorm(p_ex,p_new)]
        ufile_pvd << u_new
        pfile_pvd << p_new
        u_old = u_new

    return (max(errs_u),max(errs_p))



if __name__ == '__main__':
    main()
