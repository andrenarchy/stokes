#!/usr/bin/env python
import math
import numpy as np
from dolfin import *
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix
from numpy import intc
# krypy: https://github.com/andrenarchy/krypy
from krypy.krypy import linsys, utils
from pyamg import smoothed_aggregation_solver
#from solver_diagnostics import solver_diagnostics # pyamg

parameters.linear_algebra_backend = "uBLAS"

def main():
    # Load mesh
    #mesh = UnitCube(8, 8, 8)
    errs_u = []
    errs_p = []
    #for i in [ 2**i for i in range(2,7) ]:
        #print('Solving stokes with i={}'.format(i))
    err_u, err_p = solve_stokes(2**2, linsolver="krypy", num_deflation_vectors=10)
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

def solve_stokes(n_unknowns, linsolver="petsc", num_deflation_vectors=0):
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
    h = 1/n_unknowns #+1? TODO

    # Define function spaces
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    L = FunctionSpace(mesh, "R", 0)
    W = MixedFunctionSpace([V,Q,L])
    w = Function(W)

    # get dof mappings of subspaces
    Vdofs = W.sub(0).collapse(mesh)[1].values()
    Qdofs = W.sub(1).collapse(mesh)[1].values()
    Ldofs = W.sub(2).collapse(mesh)[1].values()
    n_dofs = len(Vdofs) + len(Qdofs) + len(Ldofs)
    print("#dofs: %d" % n_dofs)

    t0 = 0
    tend = 1
    dt = .025

    # Initial value
    u_old = Function(V)
    u_old.interpolate(u_ex)

    # for initial vector of iterative method
    w0 = Function(W)

    # deflation vectors
    Z = np.zeros( (n_dofs,0) )
    AZ = np.zeros( (n_dofs,0) )
    Proj = None

    # Define variational problem
    (u, p, lam) = TrialFunctions(W) #, TrialFunction(Q), TrialFunction(L)
    (v, q, l) = TestFunctions(W) #, TestFunction(Q), TestFunction(L)
    Avar = inner(u,v)*dx + dt*(inner(grad(u), grad(v))*dx - div(v)*p*dx - q*div(u)*dx + lam*q*dx + p*l*dx)
    bc = DirichletBC(W.sub(0), u_ex, lambda x, bd: bd)

    # variational problem for preconditioner
    Mvar = inner(u,v)*dx + dt*inner(grad(u), grad(v))*dx + p*q*dx + lam*l*dx
    Nvar = inner(grad(p),grad(q))*dx
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
        bvar = inner(u_old,v)*dx + dt*(inner(f, v)*dx + p_ex*l*dx)

        # use initial guess that satisfies boundary conditions
        w0.vector().zero()
        bc.apply(w0.vector())
        x0 = w0.vector().array().reshape((w0.vector().size(),1))

        # solve the linear system
        if linsolver in ["petsc", "lu", "gmres"]:
            solve(Avar == bvar, w, bc, solver_parameters = {"linear_solver": linsolver})
        elif linsolver=="krypy":
            A, b = assemble_system(Avar, bvar, bc)
            A = get_csr_matrix(A)
            b = b.data().reshape((b.size(),1))

            # build preconditioner 
            # cf. "Fast iterative solvers for discrete Stokes equations", 
            # Peters, Reichelt, Reusken 2005
            if Mprec is None:
                M, _ = assemble_system(Mvar, bvar, bc)
                M = get_csr_matrix(M)
                MV = M[Vdofs,:][:,Vdofs]
                MQ = M[Qdofs,:][:,Qdofs]
                ML = M[Ldofs,:][:,Ldofs]

                N, _ = assemble_system(Nvar, bvar, bc)
                N = get_csr_matrix(N)
                NQ = N[Qdofs,:][:,Qdofs]

#                solver_diagnostics(MV,
#                       fname='solver_diagnostic_MV',
#                       definiteness='positive',
#                       symmetry='hermitian'
#                       )
#                solver_diagnostics(MQ,
#                       fname='solver_diagnostic_MQ',
#                       definiteness='positive',
#                       symmetry='hermitian'
#                       )
#                solver_diagnostics(NQ,
#                       fname='solver_diagnostic_NQ',
#                       definiteness='positive',
#                       symmetry='hermitian'
#                       )
#                return

                # TODO: pyamg is non-deterministic atm. fix it! :)
                MVamg = smoothed_aggregation_solver(MV, max_levels=25, max_coarse=50)
                MQamg = smoothed_aggregation_solver(MQ, max_levels=25, max_coarse=50)
                NQamg = smoothed_aggregation_solver(NQ, max_levels=25, max_coarse=50)
                amgtol = 1e-15
                amgmaxiter = 5

                def Prec_solve(x):
                    xV = x[Vdofs]
                    xQ = x[Qdofs]
                    xL = x[Ldofs]
                    ret = np.zeros(x.shape)
                    ret[Vdofs] = MVamg.solve(xV, maxiter=amgmaxiter, tol=amgtol).reshape(xV.shape)
                    if h**2 <= dt:
                        ret[Qdofs] =           MQamg.solve(xQ, maxiter=amgmaxiter, tol=amgtol).reshape(xQ.shape) \
                                   + (1/dt)   *NQamg.solve(xQ, maxiter=amgmaxiter, tol=amgtol).reshape(xQ.shape)
                    else:
                        ret[Qdofs] = (h**2/dt)*MQamg.solve(xQ, maxiter=amgmaxiter, tol=amgtol).reshape(xQ.shape) \
                                   + (1/dt)   *NQamg.solve(xQ, maxiter=amgmaxiter, tol=amgtol).reshape(xQ.shape)
                    ret[Ldofs] = xL
                    return ret
                Prec = LinearOperator(A.shape, Prec_solve)

            # prepare deflation vectors
            if Z.shape[1] > 0:
                #Z, _ = np.linalg.qr(Z)
                AZ = A*Z
                Proj, x0 = utils.get_projection(b, Z, AZ, x0)

            itsol = linsys.minres(A, b, x0=x0, tol=1e-12, maxiter=100, Mr=Proj, M=Prec, return_basis = True)

            print("MINRES performed %d iterations with final res %e." % (len(itsol["relresvec"])-1, itsol["relresvec"][-1]) )
            w.vector().set_local(itsol["xk"])

            # extract deflation data
            if ('Vfull' in itsol) and ('Hfull' in itsol):
                if num_deflation_vectors > 0:
                    ritz_vals, ritz_coeffs, ritz_res_norm = utils.ritzh(itsol['Vfull'], itsol['Hfull'], Z, AZ, A, M=Prec)
                    sorti = np.argsort(abs(ritz_vals))
                    selection = sorti[:num_deflation_vectors]
                    nZ = Z.shape[1]
                    Z = np.dot(Z, ritz_coeffs[0:nZ, selection]) \
                      + np.dot(itsol['Vfull'][:,0:-1], ritz_coeffs[nZ:,selection])
                else:
                    Z = np.zeros( (n_dofs,0) )
        else:
            raise RuntimeError("Linear solver '%s' unknown." % linsolver)

        # Get sub-functions
        u_new, p_new, lam_new = w.split()
        errs_u += [errornorm(u_ex,u_new)]
        errs_p += [errornorm(p_ex,p_new)]
        ufile_pvd << u_new
        pfile_pvd << p_new
        u_old = u_new

    return (max(errs_u),max(errs_p))



if __name__ == '__main__':
    main()
