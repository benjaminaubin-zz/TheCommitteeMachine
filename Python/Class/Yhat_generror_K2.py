import paths
from libraries import *

class Yhat_generror_K2(object):
	def __init__(self,K=2,omega=0,V_inv=0):
		self.K = K
		self.omega = omega
		self.V_inv = V_inv  
		self.borne_inf, self.borne_sup = -10 , 10


	def H(self,x):
		res = 1/2 * erfc(x/sqrt(2))
		return res
	def integrand_gaussian_integral(self,t):
		resul = exp(-t**2/2)/sqrt(2*pi)
		return resul	
	
	# K=2 Zout / ddZout
	def Zout_K2_normalized(self):
		V_inv = self.V_inv
		omega = self.omega
		y = self.y
		a = V_inv[0,0]
		b = V_inv[1,1]
		c = V_inv[1,0]

		d_sqrt = sqrt(det(V_inv))
		#d = sqrt(a*b-c**2)
		omega2_tilde = - omega[1] * d_sqrt /sqrt(a)
		omega1_tilde = - omega[0] * sqrt(a) 
		m = (c / d_sqrt)

		N = (2*pi)**(self.K/2) / d_sqrt 

		if y == 1 :
			borne_s = self.borne_sup
			borne_i = omega2_tilde
			I =  (quad(self.Zout_K2_annex, borne_i, borne_s, args=(omega1_tilde,m,y))[0])
		if y == -1 : 
			borne_s = omega2_tilde
			borne_i = self.borne_inf
			I = (quad(self.Zout_K2_annex, borne_i, borne_s, args=(omega1_tilde,m,y))[0])
		if y == 0 : 
			borne_s = omega2_tilde
			borne_i = self.borne_inf
			I1 = (quad(self.Zout_K2_annex, borne_i, borne_s, args=(omega1_tilde,m,1))[0])
			borne_s = self.borne_sup
			borne_i = omega2_tilde
			I2 = (quad(self.Zout_K2_annex, borne_i, borne_s, args=(omega1_tilde,m,-1))[0])
			I = I1 + I2
		Zout = N * I / sqrt( (2*pi)**(self.K) / det(self.V_inv)  )
		return Zout
	def Zout_K2_annex(self,t,omega1_tilde,m,y):
		z = m*t + omega1_tilde
		gaussian_measure = exp(-t**2/2)/sqrt(2*pi)
		if y == 1 : 
			resul = gaussian_measure *(self.H(z))
		if y == -1 : 
			resul = gaussian_measure *(1-self.H(z))
		return resul

	def Yhat(self):
		norm = sqrt(det(self.V_inv)) /  ( (2*pi)**(self.K/2) )		
		Yhat = 0
		if self.K == 2 :
			for y in [+1,-1]:
				self.y = y
				Zout = y * self.Zout_K2_normalized()
				Yhat += Zout 
		return Yhat
