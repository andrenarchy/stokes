#!/usr/bin/env python
from lirk import *
import unittest
from dolfin import *

class LIRKTest(unittest.TestCase):
	'Tests for Linearly Implicit Runge-Kutta methods'
	
	def test_heat_1d(self):
		'All methods should show full convergence order for a simple heat equation example.'
		
		set_log_active(False)

		u_ex = Expression("sin(x[0])*sin(t)",t=0)

		f = Expression("sin(x[0])*(cos(t)+sin(t))",t=0)
		mesh = UnitInterval(100)
		W = FunctionSpace(mesh,"Lagrange",1)


		t0 = 0
		tend = 1

		U = TrialFunction(W)
		V = TestFunction(W)
		bc = DirichletBC(W, u_ex, lambda x, bd: bd)


		err_l8 = []
		err_l2 = []
		for dt in [ 1./2**i for i in range(3,8)]:
			a = inner(U,V)*dx + dt*inner(nabla_grad(U),nabla_grad(V))*dx

			errs = []
			u = Function(W)
			# Starting value:
			u_old = Constant(0)
			for t in np.arange(t0,tend,dt):
				# set time in expressions
				u_ex.t = t+dt
				f.t = t+dt

				# update right hand side for implicit Euler
				L = u_old*V*dx + dt*f*V*dx

				# solve the linear system
				solve(a == L, u, bc)
				#, solver_parameters = {"linear_solver": linsolver})

				errs += [errornorm(u_ex,u)]

				u_old = u

			err_l8.append(max(errs))
			err_l2.append(sqrt(np.sum(np.array(errs)**2)/len(errs)))

		err_l8 = np.array(err_l8)
		err_l2 = np.array(err_l2)
		eoc_l8 = np.log(err_l8[1:]/err_l8[:-1])/np.log(0.5)
		eoc_l2 = np.log(err_l2[1:]/err_l2[:-1])/np.log(0.5)


		# EOCs should be correct within 5%
		for eoc in eoc_l8+eoc_l2:
			self.assertGreater(eoc,1*.95)


if __name__ == '__main__':
	unittest.main()







if __name__ == '__main__':
    main()
