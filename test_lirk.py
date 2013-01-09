#!/usr/bin/env python
import math
from lirk import *
import unittest
from dolfin import *

class LIRKTest(unittest.TestCase):
	'Tests for Linearly Implicit Runge-Kutta methods'

	def test_integration(self):
		'Very simple test where the RHS does not actually depend on u should get correct EOCs.'

		t0 = 1
		tend = 2

		# Initial value:
		u0 = np.array([0.])


		def sys(v,fac,t,rhs,fac2):
			v[:] = fac*rhs[:]*fac2
		def F(v,fac,t,u):
			v[:] += fac*.1*math.exp(-.1*(t-1.))
		def dtF(v,fac,t,u):
			v[:] += fac*(-.01*math.exp(-.1*(t-1)))
		def M(v,fac,t,u):
			v[:] += fac*u[:]
		def vecsrc(n):
			l = []
			for i in range(n):
				l.append(np.array([0.]))
			return l
		def veccpy(u,v):
			v[:] = u[:]
		def zero(u):
			u[:] = 0
		def axpy(x,a,y):
			x[:] = x[:] + a*y[:]

		def u_ex(t):
			return 1.-math.exp(-.1*(t-1.))

		for scheme in (li_euler,ros2,ros3p,ros3pw,ros34pw2,ros34pw3,rowdaind2):

			err_l8 = []
			err_l2 = []
			for dt in [1./2**i for i in range(3,8)]:
				errs = []
				t = t0
				u = u0.copy()
				while t < tend:
					scheme.step(sys,F,dtF,M,t,dt,u,vecsrc,veccpy,zero,axpy)
					t += dt
					errs += [abs(u[0] - u_ex(t))]

				err_l8.append(max(errs))
				err_l2.append(sqrt(np.sum(np.array(errs)**2)/len(errs)))

			err_l8 = np.array(err_l8)
			err_l2 = np.array(err_l2)
			eoc_l8 = np.log(err_l8[1:]/err_l8[:-1])/np.log(0.5)
			eoc_l2 = np.log(err_l2[1:]/err_l2[:-1])/np.log(0.5)

			#print('Now testing scheme {}, order should be {}'.format(scheme.name,scheme.order))
			#print(eoc_l2)
			for e in eoc_l2:
				# EOCs should be reached up to 5% error
				self.assertGreater(e,scheme.order*0.95)

	def test_scalar(self):
		'Simple scalar test equation should get correct EOCs.'

		t0 = 0
		tend = 1
		lmbda = -.5

		# Initial value:
		u0 = np.array([1.])


		# Remember this is u solving (1/fac2 - J)u = rhs
		def sys(v,fac,t,rhs,fac2):
			v[:] += fac* rhs[:]/(1/fac2-lmbda)
		def F(v,fac,t,u):
			v[:] += fac*lmbda*u
		def dtF(v,fac,t,u):
			pass
		def M(v,fac,t,u):
			v[:] += fac*u[:]
		def vecsrc(n):
			l = []
			for i in range(n):
				l.append(np.array([0.]))
			return l
		def veccpy(u,v):
			v[:] = u[:]
		def zero(u):
			u[:] = 0
		def axpy(x,a,y):
			x[:] = x[:] + a*y[:]

		def u_ex(t):
			return math.exp(lmbda*t)

		for scheme in (li_euler,ros2,ros3p,ros3pw,ros34pw2,ros34pw3,rowdaind2):

			err_l8 = []
			err_l2 = []
			for dt in [1./2**i for i in range(3,8)]:
				errs = []
				t = t0
				u = u0.copy()
				while t < tend:
					scheme.step(sys,F,dtF,M,t,dt,u,vecsrc,veccpy,zero,axpy)
					t += dt
					errs += [abs(u[0] - u_ex(t))]

				err_l8.append(max(errs))
				err_l2.append(sqrt(np.sum(np.array(errs)**2)/len(errs)))

			err_l8 = np.array(err_l8)
			err_l2 = np.array(err_l2)
			eoc_l8 = np.log(err_l8[1:]/err_l8[:-1])/np.log(0.5)
			eoc_l2 = np.log(err_l2[1:]/err_l2[:-1])/np.log(0.5)

			#print('Now testing scheme {}, order should be {}'.format(scheme.name,scheme.order))
			#print(eoc_l2)
			for e in eoc_l2:
				# EOCs should be reached up to 5% error
				self.assertGreater(e,scheme.order*0.95)

	
	# Not yet fully implemented:
	@unittest.expectedFailure
	def test_heat_1d(self):
		'All methods should show full convergence order for a simple heat equation example.'
		
		set_log_active(False)

		u_ex = Expression("sin(x[0])*sin(t)",t=0)
		dt_u_ex = Expression("sin(x[0])*cos(t)",t=0)

		f = Expression("sin(x[0])*(cos(t)+sin(t))",t=0)
		dtf = Expression("sin(x[0])*(-sin(t)+cos(t))",t=0)
		mesh = UnitInterval(100)
		W = FunctionSpace(mesh,"Lagrange",1)


		t0 = 0
		tend = 1

		U = TrialFunction(W)
		V = TestFunction(W)

		scheme = ros2
		S = assemble(inner(nabla_grad(U),nabla_grad(V))*dx)
		mass = assemble(U*V*dx)

		err_l8 = []
		err_l2 = []
		for dt in [ 1./2**i for i in range(3,8)]:
			a = 1./scheme.gamma_diag/dt*U*V*dx + inner(nabla_grad(U),nabla_grad(V))*dx
			A = assemble(a)

			errs = []
			# Starting value:
			u = Function(W)
			# Unnecessary, but if we had more complex initial values...
			# u.interpolate(Constant(0))

			def F(v,fac,t,u):
				f.t = t
				L = f*V*dx
				b = assemble(L)
				b -= S*u
				v[:] += fac*b
			def dtF(v,fac,t,u):
				dtf.t = t
				b = assemble(dtf*V*dx)
				v[:] += fac*b
			def M(v,fac,t,u):
				v[:] += fac*mass*u
			def sys(v,fac,t,rhs,fac2):
				dt_u_diri.fac = fac2
				dt_u_diri.t = t
				bc.apply(A,rhs)
				x = Function(W)
				solve(A,x.vector(),rhs)
				v[:] += fac*x.vector()
			def vecsrc(n):
				l = []
				for i in range(n):
					l.append(Function(W).vector())
				return l
			def veccpy(u,v):
				v[:] = u[:]
			def zero(u):
				u[:] = 0
			def fenics_axpy(x,a,y):
				x.axpy(a,y)


			t = t0
			bc = DirichletBC(W, dt_u_ex, lambda x, bd: bd)
			dt_u_diri = Expression("fac*sin(x[0])*cos(t)",fac=0,t=0)
			bc = DirichletBC(W, dt_u_diri, lambda x, bd: bd)

			while t < tend:
				t += dt

				scheme.step(sys,F,dtF,M,t,dt,u.vector(),vecsrc,veccpy,zero,fenics_axpy)

				u_ex.t = t+dt
				errs += [errornorm(u_ex,u)]
			#ue = Function(W)
			#ue.interpolate(u_ex)
			#plot(u-ue)
			#interactive()


			err_l8.append(max(errs))
			err_l2.append(sqrt(np.sum(np.array(errs)**2)/len(errs)))

		err_l8 = np.array(err_l8)
		err_l2 = np.array(err_l2)
		eoc_l8 = np.log(err_l8[1:]/err_l8[:-1])/np.log(0.5)
		eoc_l2 = np.log(err_l2[1:]/err_l2[:-1])/np.log(0.5)

		print(err_l2)
		print(eoc_l2)
		print(err_l8)
		print(eoc_l8)


		# EOCs should be correct within 5%
		for eoc in eoc_l8:
			self.assertGreater(eoc,scheme.order*.95)
		for eoc in eoc_l2:
			self.assertGreater(eoc,scheme.order*.95)


if __name__ == '__main__':
	unittest.main()







if __name__ == '__main__':
    main()
