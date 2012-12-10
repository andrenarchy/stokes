#!/usr/bin/env python3
# -*- coding: utf8 -*-
'''Time discretization by Rosenbrock methods.

Author: Stephan Weller <weller@math.fau.de>
Date:   2012/12/08

This module realizes \f$s\f$-stage Linearly Implicit Runge-Kutta (LIRK) methods.'''

import numpy as np
from math import sqrt
from numpy.linalg import inv



class LIRK():
	'''Linearly Implicit Runge-Kutta (LIRK) method.

	Such a method is defined by the coefficients
	\f$\alpha_{ij}, \gamma_{ij}, b_i, i,j=1,\dots,s\f$ as follows:

	For the initival value problem \f[M\frac{d}{dt} u = F(t,u)\f]
	we iterate the steps for \f$i=1,\dots,s\f$:

	\f[ (M-\tau\gamma_{ii}\partial_u F(t_n,u_n)) K_{ni} =
	F(t_n+\alpha_i\tau,u_n+\tau\sum_{j=1}^{i-1} \alpha_{ij}K_{nj})
	+ \tau\partial_u F(t_n,u_n)\sum_{j=1}^{i-1}\gamma_{ij}K_{nj}
	+ \tau\gamma_i \partial_t F(t_n,u_n)\f]

	\f[ u_{n+1} = u_n + \tau\sum_{i=1}^s b_i K_{ni} \f]

	where \f$\alpha_i = \sum_{j=1}^{i-1} \alpha_{ij},
				\gamma_i = \sum_{j=1}^i \gamma_{ij}\f$.

	We require the method to be diagonal, i.e. \f$\gamma_{ii} = \gamma\f$ for all \f$i=1,\dots,s\f$.
	This allows for a more efficient implementation, as the system matrix has to
	be assembled but once per time step.

	An embedded method (usually of lower order) may be given by using some
	coefficients \f$\hat b\f$ instead of \f$b\f$.

	For more efficient programming, one can use the following transformed
	scheme (the coefficients \f$a_{ij},c_{ij},m_i\f$ are computed automatically from the original
	coefficents):

	\f[\left(\frac{M}{\tau\gamma} -\partial_u F(t_n,u_n)\right)U_{ni} =
	F(t_n+\alpha_i\tau,u_n+\sum_{j=1}^{i-1}a_{ij}U_{nj})
	+ \sum_{j=1}^{i-1} \frac{c_{ij}}{\tau}M U_{nj} + \tau\gamma_i \partial_t F(t_n,u_n), i=1,\dots, s\f]

	\f[u_{n+1} = u_n + \sum_{i=1}^s m_i U_{ni} \f]

	*Note:* The matrix \f$M\f$ is not required to be invertible, i.e.
	differential algebraic equations can also be treated.'''

	def __init__(self,num_stages,alpha,gamma,b,bhat=None):
		'''Initialize a Butcher Tableau for a Rosenbrock method

		Keyword arguments:

		self				-- the LIRK method
		num_stages  -- number of stages \f$s\f$ in the method
		alpha       -- The coefficients \f$\alpha_{ij}, i,j=1,\dots s\f$
		gamma       -- The coefficients \f$\gamma_{ij}, i,j=1,\dots s\f$
		b           -- The coefficients \f$b_i, i=1,\dots,s\f$
		bhat        -- The coefficients \f$\hat b_i, i=1,\dots,s\f$ for an embedded method of lower order'''

		# Brace yourselves, matrix multiplications are coming
		self.num_stages = num_stages
		self.alpha = np.matrix(alpha)
		self.gamma = np.matrix(gamma)
		self.b = np.array(b)
		self.bhat = np.array(bhat)

		# Compute convenience variables
		self.gamma_diag = self.gamma[0,0]
		self.alpha_i = np.sum(self.alpha,axis=1)
		self.gamma_i = np.sum(self.gamma,axis=1)

		# Compute transformed variables
		self.c = -inv(self.gamma)
		self.a = -self.alpha*self.c
		self.m = -self.b*self.c
		if self.bhat.any():
			self.mhat = -self.bhat*self.c


	def check(self):
		'''Checks the coefficients of a butcher tableau

		The tableau is checked for order conditions, PDE suitability,
		Usability for DAEs, stability, and inexact Jacobians'''

		def eprint(string,err):
			print('\t{}: {}'.format(string,'✔' if err < 1e-10 else '✘'))

		# Everything as array as we need pointwise operations (.* etc.)
		alpha = np.asarray(self.alpha)
		gamma = np.asarray(self.gamma)
		b = np.asarray(self.b)
		beta = np.asarray(alpha + gamma)
		alphai = np.asarray(np.sum(np.tril(alpha,-1),axis=1))
		betai_sub = np.asarray(np.sum(np.tril(beta,-1),axis=1))
		betai = np.asarray(np.sum(np.tril(beta),axis=1))
		gammai = np.asarray(np.sum(np.tril(gamma,-1),axis=1))
		omega = np.asarray(inv(beta))

		print('\tTesting {}-stage LIRK method.'.format(self.num_stages))
		err = abs(np.sum(b)-1)
		eprint('Order 1',err)
		err += abs(np.sum(b*betai) - 1/2)
		eprint('Order 2',err)
		err = err + abs(np.sum(b*alphai**2) - 1/3)
		err2 = -1/6
		for i in range(self.num_stages):
			err2 += np.sum(b[i]*beta[i,:]*betai)
		err += abs(err2)
		eprint('Order 3', err)
		if self.num_stages >= 3:
			err2 = -1
			for i in range(self.num_stages):
				err2 += np.sum(self.b[i]*omega[i,:]*alphai[:])
			err = err + abs(err2)
			eprint('Order 3 for DAEs of order 1',err)

		if self.num_stages == 3:
			err = self.b[2]*(self.alpha[2,1]+self.gamma[2,1])*np.sum(self.alpha[1,:1])**2 - 1/6 + 2/3*self.gamma_diag
			err2 = self.gamma_diag - .5 - 1/6*sqrt(3)
			eprint('PDE suitable', abs(err)+abs(err2))
		elif self.num_stages == 4:
			err = self.b[1]*alphai[1]+self.b[2]*alphai[2]+self.b[3]*alphai[3] - 1/2
			eprint('Consistency for O(dt)-Jacobian approximation',abs(err))
			err = self.b[2]*self.alpha[2,1]*alphai[1] + \
				self.b[3]*(self.alpha[3,1]*alphai[1] + self.alpha[3,2]*alphai[2]) - 1/6
			err2 = self.b[2]*self.alpha[2,1]*betai_sub[1] + \
				self.b[3]*(self.alpha[3,1]*betai_sub[1] + self.alpha[3,2]*betai_sub[2]) - 1/6 + self.gamma_diag/2
			err3 = self.b[2]*beta[2,1]*alphai[1] + \
				self.b[3]*(beta[3,1]*alphai[1] + beta[3,2]*alphai[2]) - 1/6 + self.gamma_diag/2
			eprint('Order 3 for arbitrary W-matrices',abs(err)+abs(err2)+abs(err3))

			err = self.b[3]*beta[2,1]*beta[3,2]*alphai[1]**2 \
				- 2*self.gamma_diag**4 + 2*self.gamma_diag**3 - 1/3*self.gamma_diag**2
			err2 = self.b[2]*beta[2,1]*alphai[1]**2 + \
				self.b[3]*(beta[3,1]*alphai[1]**2+beta[3,2]*alphai[2]**2) \
				- 2*self.gamma_diag**3 + 3*self.gamma_diag**2 - 2/3*self.gamma_diag
			err3 = self.b[3]*beta[3,2]*beta[2,1]*beta[1,0]
			eprint('PDE suitability',abs(err)+abs(err2)+abs(err3))

			err = self.gamma_diag**4 - 3*self.gamma_diag**3+3/2*self.gamma_diag**2-1/6*self.gamma_diag
			eprint('L-stability',abs(err))

	def __repr__(self):
		res = 'LIRK('
		res += 'num_stages={},'.format(self.num_stages)
		res += 'alpha={},'.format(self.alpha)
		res += 'gamma={},'.format(self.gamma)
		res += 'b={},'.format(self.b)
		res += 'bhat={},'.format(self.bhat) if self.bhat.any() else ''
		res += ')'
		return res

	def __str__(self):
		res = 'LIRK method of stage order {}\n'.format(self.num_stages)
		res += 'alpha:\n{}\n'.format(self.alpha)
		res += 'gamma:\n{}\n'.format(self.gamma)
		res += 'b:\n{}\n'.format(self.b)
		res += 'bhat:\n{}\n'.format(self.bhat) if self.bhat.any() else ''
		return res

	def step(self,sys,F,dtF,M,t,dt,u,u_new,err_est):
		'''Executes one LIRK step

		For the operators, you have to define the operations:
		- \f$y += fac\cdot F(u,t)\f$, the routine `F`
		- \f$y += fac\cdot \partial_t F(u,t)\f$, the routine `dtF`
		- \f$y += fac\cdot M(t)u\f$, the routine `mass`
		- \f$y += \left(\frac{M}{fac}-\partial_u F\right)^{-1}u\f$, the routine `sys`

		Keyword arguments:

		self				-- the LIRK method
		sys   			-- system matrix solving routine
		F     			-- right-hand side \f$F\f$
		dtF   			-- right-hand side time derivative \f$\partial_t F\f$
		M		  			-- left-hand side operator \f$M\f$
		t     			-- time \f$t_n\f$
		dt    			-- time step size \f$\tau\f$
		u     			-- vector from old time step \f$u_n\f$
		u_new 			-- computed solution \f$u_{n+1}\f$
		err_est 		-- error estimator computed from difference to embedded lower order method.'''

		for i in range(self.num_stages):
			ti = t + self.alpha_i[i]*dt
			ui = u
			for j in range(i-1):
				ui += self.a[i,j]*U_ni[:,j]
			rhs = np.zeros(np.size(u))
#      call F(ui,rhs,ti,1d0)
			for j in range(i-1):
				if self.c[i,j]:
					pass
#          call mass(U_ni(:,j),rhs(:),ti,B%c(i,j)/dt)
			if self.gamma_i[i]:
				pass
#        call dtF(u(:),rhs(:),t,dt*B%gamma_i(i))
			U_ni[:,i] = 0
#      call sys(rhs(:),U_ni(:,i),t,dt*B%gamma_diag)

		#TODO these should just copy, not change references
		u_new = u
		err_est = 0
		for i in range(self.num_stages):
			u_new += self.m[i]*U_ni[:,i]
			if self.bhat.any():
				err_est += self.mhat[i]*U_ni[:,i] - self.m[i]*U_ni[:,i]


# Linearly implicit Euler method
# see any Numerics book, e.g. Deuflhard/Bornemann, Numerische Mathematik 2, 3. Auflage, p. 293.
li_euler = LIRK(1,[[0]],[[1]],[1])

# ROS2
# see K. Dekker, J. G. Verwer: Stability of Runge-Kutta Methods for Stiff
# Nonlinear Differential Equations, Elsevier -- North Holland, Amsterdam, 1984
gamma = 2.928932188134e-1
alpha_ros2 = [[0,0]]
alpha_ros2.append([1,0])
gamma_ros2 = [[gamma,0]]
gamma_ros2.append([-5.857864376269e-1,gamma])

ros2 = LIRK(2,alpha_ros2,gamma_ros2,[.5,.5],[.5,.5])

del(gamma,alpha_ros2,gamma_ros2)

# ROS3P
# see J. Lang, J.Verwer: ROS3P -- An accurate third-order Rosenbrock solver
# designed for parabolic problems. Bit Numerical Mathematics, Vol. 41, No. 4,
# pp. 731--738, 2001, http://www3.mathematik.tu-darmstadt.de/fileadmin/home/groups/10/Paper_Lang/BIT01-LangVerwer.pdf
gamma = .5+sqrt(3)/6
alpha_ros3p = [[0,0,0]]
alpha_ros3p.append([1,0,0])
alpha_ros3p.append([1,0,0])
gamma_ros3p = [[gamma,0,0]]
gamma_ros3p.append([-1,gamma,0])
gamma_ros3p.append([-gamma,.5-2*gamma,gamma])
b_ros3p = [2/3, 0, 1/3]
bhat_ros3p = [1/3, 1/3, 1/3]

ros3p = LIRK(3,alpha_ros3p,gamma_ros3p,b_ros3p,bhat_ros3p)

del(gamma,alpha_ros3p,gamma_ros3p,b_ros3p,bhat_ros3p)

#  ROS3PW
#  see J. Rang, L. Angermann: New Rosenbrock W-methods of order 3 for partial
#  differential algebraic equations of index 1. Bit Numerical Mathematics,
#  Vol. 45, pp. 761--787, 2005, [doi: 10.1007/s10543-005-0035-y](http://dx.doi.org/10.1007/s10543-005-0035-y)
gamma = .5+sqrt(3)/6
alpha_ros3pw = [[0,0,0]]
alpha_ros3pw.append([1.5773502691896257,0,0])
alpha_ros3pw.append([.5,0,0])
gamma_ros3pw = [[gamma,0,0]]
gamma_ros3pw.append([-1.5773502691896257, gamma, 0])
gamma_ros3pw.append([-.67075317547305480, -.17075317547305482,gamma])
b_ros3pw = [.10566243270259355, .049038105676657971, .84529946162074843]
bhat_ros3pw = [ -.1786327949540818, 1/3, .84529946162074843]

ros3pw = LIRK(3,alpha_ros3pw,gamma_ros3pw,b_ros3pw,bhat_ros3pw)

del(gamma,alpha_ros3pw,gamma_ros3pw,b_ros3pw,bhat_ros3pw)

# ROS34PW2
# see J. Rang, L. Angermann: New Rosenbrock W-methods of order 3 for partial
# differential algebraic equations of index 1. Bit Numerical Mathematics,
# Vol. 45, pp. 761--787, 2005, [doi: 10.1007/s10543-005-0035-y](http://dx.doi.org/10.1007/s10543-005-0035-y)
gamma  =  .43586652150845900
a21    =  .87173304301691801
a31    =  .84457060015369423
a32    = -.11299064236484185
a41    =  0
a42    =  0
a43    =  1
c21    = -.87173304301691801
c31    = -.90338057013044082
c32    =  .054180672388095326
c41    =  .24212380706095346
c42    =-1.2232505839045147
c43    =  .54526025533510214

alpha_ros34pw2 = [[0,0,0,0]]
alpha_ros34pw2.append([a21, 0, 0, 0])
alpha_ros34pw2.append([a31, a32, 0, 0])
alpha_ros34pw2.append([a41, a42, a43, 0])
gamma_ros34pw2 = [[gamma,0,0,0]]
gamma_ros34pw2.append([c21, gamma, 0, 0])
gamma_ros34pw2.append([c31, c32, gamma, 0])
gamma_ros34pw2.append([c41, c42, c43, gamma])
b_ros34pw2 = [c41, c42, 1+c43,.43586652150845900]
bhat_ros34pw2 = [.37810903145819369, -.096042292212423178, .5, .21793326075422950]

ros34pw2 = LIRK(4,alpha_ros34pw2,gamma_ros34pw2,b_ros34pw2,bhat_ros34pw2)

del(gamma,a21,a31,a32,a41,a42,a43,c21,c31,c32,c41,c42,c43,alpha_ros34pw2,gamma_ros34pw2,b_ros34pw2,bhat_ros34pw2)

# ROS34PW3
# see J. Rang, L. Angermann: New Rosenbrock W-methods of order 3 for partial
# differential algebraic equations of index 1. Bit Numerical Mathematics,
# Vol. 45, pp. 761--787, 2005, [doi: 10.1007/s10543-005-0035-y](http://dx.doi.org/10.1007/s10543-005-0035-y)
gamma  = 1.0685790213016289
a21    = 2.5155456020628817
a31    = 5.0777280103144085e-1
a32    = 7.5e-1
a41    = 1.3959081404277204e-1
a42    =-3.3111001065419338e-1
a43    = 8.2040559712714178e-1
c21    =-2.5155456020628817
c31    =-8.7991339217106512e-1
c32    =-9.6014187766190695e-1
c41    =-4.1731389379448741e-1
c42    = 4.1091047035857703e-1
c43    =-1.3558873204765276

alpha_ros34pw3 = [[ 0, 0, 0, 0]]
alpha_ros34pw3.append([a21, 0, 0, 0])
alpha_ros34pw3.append([a31, a32, 0, 0])
alpha_ros34pw3.append([a41, a42, a43, 0])
gamma_ros34pw3 = [[ gamma, 0, 0, 0]]
gamma_ros34pw3.append([c21, gamma, 0, 0])
gamma_ros34pw3.append([c31, c32, gamma, 0])
gamma_ros34pw3.append([c41, c42, c43, gamma])
b_ros34pw3 = [2.2047681286931747e-1, 2.7828278331185935e-3, 7.1844787635140066e-3, 7.6955588053404989e-1]
bhat_ros34pw3 = [ 3.1300297285209688e-1, -2.8946895245112692e-1, 9.7646597959903003e-1, 0 ]

ros34pw3 = LIRK(4,alpha_ros34pw3,gamma_ros34pw3,b_ros34pw3,bhat_ros34pw3)

del(gamma,a21,a31,a32,a41,a42,a43,c21,c31,c32,c41,c42,c43,alpha_ros34pw3,gamma_ros34pw3,b_ros34pw3,bhat_ros34pw3)

# ROWDAIND2
# see Ch. Lubich, M. Roche: Rosenbrock methods for Differential-algebraic
# Systems with Solution-dependent Singular Matrix Multiplying the Derivative.
# Computing, Vol. 43, pp. 325--342, 1990, [doi: !10.1007/BF02241653](http://dx.doi.org/10.1007/BF02241653)
gamma  = .3
a21    = 5e-1
a31    = 2.8e-1
a32    = 7.2e-1
a41    = 2.8e-1
a42    = 7.2e-1
a43    = 0
c21    = -1.121794871794876e-1
c31    = 2.54
c32    = -3.84
c41    = 29/75
c42    =-7.2e-1
c43    = 1/30

alpha_rowdaind2 = [[ 0, 0, 0, 0]]
alpha_rowdaind2.append([a21, 0, 0, 0])
alpha_rowdaind2.append([a31, a32, 0, 0])
alpha_rowdaind2.append([a41, a42, a43, 0])
gamma_rowdaind2 = [[ gamma, 0, 0, 0]]
gamma_rowdaind2.append([c21, gamma, 0, 0])
gamma_rowdaind2.append([c31, c32, gamma, 0])
gamma_rowdaind2.append([c41, c42, c43, gamma])
b_rowdaind2 = [ 2/3, 0, 1/30, .3 ]
bhat_rowdaind2 = [ 4.799002800355166e-1, 5.176203811215082e-1, 2.479338842975209e-3, 0]

rowdaind2 = LIRK(4,alpha_rowdaind2,gamma_rowdaind2,b_rowdaind2,bhat_rowdaind2)

del(gamma,a21,a31,a32,a41,a42,a43,c21,c31,c32,c41,c42,c43,alpha_rowdaind2,gamma_rowdaind2,b_rowdaind2,bhat_rowdaind2)



def main():
	print('LI Euler')
	li_euler.check()
	print('ROS2')
	ros2.check()
	print('ROS3P')
	ros3p.check()
	print('ROS3PW')
	ros3pw.check()
	print('ROS34PW2')
	ros34pw2.check()
	print('ROS34PW3')
	ros34pw3.check()
	print('ROWDAIND2')
	rowdaind2.check()

if __name__ == '__main__':
	main()
