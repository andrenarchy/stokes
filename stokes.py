#!/usr/bin/env python
import math
import numpy as np
from dolfin import *
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix, bmat
from numpy import intc
# krypy: https://github.com/andrenarchy/krypy
from krypy.krypy import linsys
from pyamg import smoothed_aggregation_solver

parameters.linear_algebra_backend = "uBLAS"

def main():
    # Load mesh
    #mesh = UnitCube(8, 8, 8)
    errs_u = []
    errs_p = []
    #for i in [ 2**i for i in range(2,7) ]:
        #print('Solving stokes with i={}'.format(i))
    err_u, err_p = solve_stokes(2**4, linsolver="krypy")
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

def get_csr_matrix(A):
    '''get csr matrix from dolfin without copying data

    cf. http://code.google.com/p/pyamg/source/browse/branches/2.0.x/Examples/DolfinFormulation/demo.py
    '''
    (row,col,data) = A.data()
    return csr_matrix( (data,intc(col),intc(row)), shape=(A.size(0),A.size(1)) )

def getLinearOperator(A):
    '''construct a linear operator for easy application in a Krylov subspace method

    In a Krylov subspace method we only need the application of a linear operator
    to a vector or a block and this function returns a scipy.sparse.linalg.LinearOperator 
    that just does this.
    '''
    def matvec(v):
        vvec = Vector(A.size(1))
        vvec.set_local(v.reshape(v.shape[0]))
        resvec = Vector(A.size(0))
        A.mult(vvec, resvec)
        return resvec.array()
    return LinearOperator( (A.size(0), A.size(1)), matvec=matvec )

def solve_stokes(n_unknowns, linsolver="petsc"):
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
    
    u_new = Function(V)
    p_new = Function(Q)
    lam_new = Function(L)

    t0 = 0
    tend = 1
    dt = .05

    # Initial value
    u_old = Constant((0,0))

    # Define variational problem
    (u, p, lam) = TrialFunction(V), TrialFunction(Q), TrialFunction(L)
    (v, q, l) = TestFunction(V), TestFunction(Q), TestFunction(L)
    bc = DirichletBC(V, u_ex, lambda x, bd: bd)

    # for mass matrix
    Mvar = inner(u,v)*dx
    # for stiffness matrix
    Svar = inner(grad(u), grad(v))*dx
    # for pressure
    Bvar = div(v)*p*dx
    # for velocity divergence
    Btvar = q*div(u)*dx
    # lagrange multiplier for divergence
    Evar = lam*q*dx
    Etvar = p*l*dx
    # for pressure mass matrix (needed in preconditioner)
    Mpvar = p*q*dx

    M = assemble(Mvar)
    bc.apply(M)
    M = get_csr_matrix(M)

    S = assemble(Svar)
    bc.zero(S)
    S = get_csr_matrix(S)

    B = assemble(Bvar)
    bc.zero(B)
    B = get_csr_matrix(B)

    Bt = get_csr_matrix(assemble(Btvar))
    E = get_csr_matrix(assemble(Evar))
    Et = get_csr_matrix(assemble(Etvar))
    A = bmat( [[M + dt*S, -dt*B,  None],
               [-dt*Bt,    None,  dt*E],
               [None,     dt*Et, None]], format='csr')

    # variational problem for preconditioner
    # TODO: adapt preconditioner to time-dependent setting
    #Mvariational = inner(grad(u), grad(v))*dx + p*q*dx
    Mprec = None


    ufile_pvd = File("velocity.pvd")
    pfile_pvd = File("pressure.pvd")
    errs_u = []
    errs_p = []
    for t in np.arange(t0,tend,dt):
        # set time in expressions
        u_ex.t = t+dt
        p_ex.t = t+dt
        f.t = t+dt

        # update right hand side for implicit Euler
        bvvar = inner(u_old,v)*dx + dt*inner(f, v)*dx
        bqvar = Constant(0.0)*q*dx
        blvar = dt*p_ex*l*dx

        bv = assemble(bvvar)
        bc.apply(bv)
        bq = assemble(bqvar)
        bl = assemble(blvar)
        b = np.r_[ bv.data(), bq.data(), bl.data()]
        b = b.reshape((b.shape[0],1))

        u_new.vector().zero()
        bc.apply(u_new.vector())
        x0 = np.r_[ u_new.vector().array().reshape((u_new.vector().size(),1)), np.zeros((bq.size(),1)), np.zeros((bl.size(),1))]

        # solve the linear system
        if linsolver in ["petsc", "lu", "gmres"]:
            solve(a == L, U, bc, solver_parameters = {"linear_solver": linsolver})
        elif linsolver=="krypy":
            #if Mprec is None:
            #    M, _ = assemble_system(Mvariational, L, bc)
            #    Mcsr = get_csr_matrix(M)
            #    Mamg = smoothed_aggregation_solver(Mcsr, max_levels=25, max_coarse=50)
            #    def Mamg_solve(x):
            #        return Mamg.solve(x, maxiter=5, tol=0.0).reshape(x.shape)
            #    Mprec = LinearOperator( (M.size(0), M.size(1)), Mamg_solve)

            itsol = linsys.minres(A, b, x0=x0, tol=1e-9)

            print("MINRES performed %d iterations with final res %e." % (len(itsol["relresvec"])-1, itsol["relresvec"][-1]) )
            u_new.vector().set_local(itsol["xk"][:bv.size()])
            p_new.vector().set_local(itsol["xk"][bv.size():bv.size()+bq.size()])
            lam_new.vector().set_local(itsol["xk"][bv.size()+bq.size():])
        else:
            raise RuntimeError("Linear solver '%s' unknown." % linsolver)


        # Get sub-functions
        errs_u += [errornorm(u_ex,u_new)]
        errs_p += [errornorm(p_ex,p_new)]
        ufile_pvd << u_new
        pfile_pvd << p_new
        u_old = u_new

    return (max(errs_u),max(errs_p))



if __name__ == '__main__':
    main()
