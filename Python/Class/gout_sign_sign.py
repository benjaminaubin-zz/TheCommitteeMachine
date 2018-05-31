import paths
from libraries import *

class gout_sign_sign(object):
	def __init__(self,K=1,y=0,omega=0,V=0,V_inv=0):
		self.K = K
		
		self.y = y 
		self.omega = omega
		self.V = V  # (Q-q)
		self.V_inv = V_inv # (Q-q)^-1
		
		if self.K > 2 :
			self.b = self.V_inv[0][1]
			self.a = self.V_inv[0][0] - self.b
		else : 
			self.b = 0
			self.a = self.V_inv[0]
		
		self.threshold_zero = 1e-8
		self.borne_inf, self.borne_sup = -10 , 10

		self.gout = np.zeros((self.K,1)) 
		self.dgout = np.zeros((self.K,self.K))

		# Options 	
		self.print_Chrono = False
		self.print_Chrono_global = False

	def H(self,x):
		res = 1/2 * erfc(x/sqrt(2))
		return res
	def integrand_gaussian_integral(self,t):
		resul = exp(-t**2/2)/sqrt(2*pi)
		return resul	
	
	# K=1 Zout / ddZout 
	def Zout_K1(self):
		V = self.V
		omega = self.omega
		y = self.y
		if y == 1 :
			Zout = sqrt(2*pi*V) * self.H(-omega/sqrt(V))
		elif y == -1 : 
			Zout = sqrt(2*pi*V) * (1-self.H(-omega/sqrt(V)))
		else :
			Zout = 0
		return Zout
	def dZout_K1(self):
		V = self.V
		omega = self.omega
		y = self.y
		if y == 1 :
			dZout =  exp(-omega**2/(2*V))
		elif y == -1 : 
			dZout =   -exp(-omega**2/(2*V))
		else :
			dZout = 0
		return dZout
	def ddZout_K1(self):
		V = self.V
		omega = self.omega
		y = self.y
		if y == 1 :
			ddZout =  - omega/sqrt(V) * exp(-omega**2/(2*V))
		elif y == -1 : 
			ddZout =   omega/sqrt(V) * exp(-omega**2/(2*V))
		else :
			ddZout = 0
		return ddZout
	
	# K=2 Zout / ddZout
	def Zout_K2(self):
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
		Zout = N * I
		return Zout
	def Zout_K2_annex(self,t,omega1_tilde,m,y):
		z = m*t + omega1_tilde
		gaussian_measure = exp(-t**2/2)/sqrt(2*pi)
		if y == 1 : 
			resul = gaussian_measure *(self.H(z))
		if y == -1 : 
			resul = gaussian_measure *(1-self.H(z))
		return resul
	def dZout_K2(self):
		V_inv , omega , y = self.V_inv , self.omega , self.y
		dZout_K2 = np.zeros(self.K)
		a , b , c = V_inv[0,0] , V_inv[1,1] , V_inv[0,1]
		d = det(V_inv)
		if y == 1 : 
			dZout_K2[0] = self.dZout_K2_annex(omega[0] , omega[1] , b ,'sup')[0]
			dZout_K2[1] = self.dZout_K2_annex(omega[1] , omega[0] , a ,'sup')[0]

		if y == -1 : 
			dZout_K2[0] = - self.dZout_K2_annex(omega[0] , omega[1] , b ,'inf')[0]
			dZout_K2[1] = - self.dZout_K2_annex(omega[1] , omega[0] , a ,'inf')[0]

		if y == 0 :
			I , N = self.dZout_K2_annex(omega[0] , omega[1] , b ,'sup')
			dZout_K2[0] = N - 2*I 
			
			I , N = self.dZout_K2_annex(omega[1] , omega[0] , a ,'sup')
			dZout_K2[1] = N - 2*I
		return dZout_K2
	def dZout_K2_annex(self,w1,w2,x,borne_infinity):
		V , V_inv = self.V , self.V_inv 
		a , b , c = V_inv[0,0] , V_inv[1,1] , V_inv[0,1]
		d = det(V_inv)
		N = sqrt(2*pi)/sqrt(x) * exp(-1/2 * w1**2 * d / x)
		borne = - w2 * sqrt(x) - c/sqrt(x) * w1
		if borne_infinity == 'sup':
			borne_inf = borne 
			borne_sup = self.borne_sup
			I = quad(self.integrand_gaussian_integral ,borne_inf ,borne_sup, args=())[0]
			res = N * I 
		else : 
			borne_inf = self.borne_inf 
			borne_sup = borne
			I = quad(self.integrand_gaussian_integral ,borne_inf ,borne_sup, args=())[0]
			res = N * I 
		return (res,N)
	def ddZout_K2_annex_1(self,w1,w2,x,c,d,borne_infinity):
		term1 = 1/sqrt(x) * exp(- 1/2 * w1**2 * d / x)
		term2 =  - w1 * d / x
		borne = - w2 * sqrt(x) - c/sqrt(x) * w1
		term3 = - exp(- 1/2 * borne**2 ) * ( -c / sqrt(x))
		if borne_infinity == 'borne_sup':
			#I = sqrt(2*pi) * quad(self.integrand_gaussian_integral ,borne ,self.borne_sup, args=())[0]
			I = sqrt(2*pi) * self.H(borne)
			res = term1 * (  term2 * I  + term3 )
		elif borne_infinity == 'borne_inf':
			I = sqrt(2*pi) * (1 -self.H(borne))
			#I = sqrt(2*pi) * quad(self.integrand_gaussian_integral ,self.borne_inf ,borne, args=())[0]
			res = term1 * (  term2 * I  - term3 )
		else : 
			print('error ddZout_K2_annex_1') 

		
		
		return res 
	def ddZout_K2_annex_2(self,w1,w2,x,c,d):
		term1 = 1/sqrt(x) * exp(- 1/2 * w1**2 * d / x)
		borne_inf = - w2 * sqrt(x) - c/sqrt(x) * w1
		term4 = - exp(- 1/2 * borne_inf**2 ) * ( - sqrt(x))
		res =  term1 * term4
		return res 
	def ddZout_K2(self):
		V_inv , omega , y = self.V_inv , self.omega , self.y

		ddZout_K2 = np.zeros((self.K,self.K))
		a , b , c = V_inv[0,0] , V_inv[1,1] , V_inv[0,1]
		d = a*b - c**2 

		if y == 1 : 
			ddZout_K2[0,0] = self.ddZout_K2_annex_1( omega[0],omega[1],b,c,d,'borne_sup')
			ddZout_K2[0,1] = self.ddZout_K2_annex_2( omega[0],omega[1],b,c,d)
			ddZout_K2[1,1] = self.ddZout_K2_annex_1( omega[1],omega[0],a,c,d,'borne_sup')
			ddZout_K2[1,0] = self.ddZout_K2_annex_2( omega[1],omega[0],a,c,d)
		if y == -1 : 
			ddZout_K2[0,0] = - self.ddZout_K2_annex_1( omega[0],omega[1],b,c,d,'borne_inf')
			ddZout_K2[0,1] = self.ddZout_K2_annex_2( omega[0],omega[1],b,c,d)
			ddZout_K2[1,1] = - self.ddZout_K2_annex_1( omega[1],omega[0],a,c,d,'borne_inf')
			ddZout_K2[1,0] = self.ddZout_K2_annex_2( omega[1],omega[0],a,c,d)
		if y == 0 : 
			ddZout_K2[0,0] = - self.ddZout_K2_annex_1( omega[0],omega[1],b,c,d,'borne_sup') + self.ddZout_K2_annex_1( omega[0],omega[1],b,c,d,'borne_inf')
			ddZout_K2[0,1] = - self.ddZout_K2_annex_2( omega[0],omega[1],b,c,d) - self.ddZout_K2_annex_2( omega[0],omega[1],b,c,d)
			ddZout_K2[1,1] =  - self.ddZout_K2_annex_1( omega[1],omega[0],a,c,d,'borne_sup') + self.ddZout_K2_annex_1( omega[1],omega[0],a,c,d,'borne_inf')
			ddZout_K2[1,0] = - self.ddZout_K2_annex_2( omega[1],omega[0],a,c,d) - self.ddZout_K2_annex_2( omega[1],omega[0],a,c,d) 

		return ddZout_K2

	def fout_gout(self):
		if self.print_Chrono_global :
			start = time.time()

		norm = sqrt(det(self.V_inv)) /  ( (2*pi)**(self.K/2) )		

		# K == 2
		if self.K == 2 :
			Zout = self.Zout_K2()
			if Zout == 0 or Zout < self.threshold_zero :
				Zout = 0
				gout = np.zeros((1,self.K))
			else :
				dZout = self.dZout_K2()
				gout = 1/Zout * dZout

		# K == 1 
		elif self.K == 1 : 
			Zout = self.Zout_K1()
			if Zout == 0 :
				gout = np.zeros((1,self.K))
			else :
				dZout = self.dZout_K1()
				gout = 1/Zout * dZout
			if np.isinf(gout):
				gout = 0 

		fout =  norm * Zout

		if self.print_Chrono_global : 
			end = time.time()
			print('Time gout_dgout=',end-start)
		
		return (fout,gout)
	def gout_dgout(self):
		if self.print_Chrono_global :	
			start = time.time()
		if self.K == 1 :
			Zout = self.Zout_K1()
			if Zout == 0 or Zout < self.threshold_zero:
				gout = np.zeros((1,self.K))
				dgout = np.zeros((self.K,self.K))
			else :
				dZout = self.dZout_K1()
				gout = 1/Zout * dZout
			 
				ddZout = self.ddZout_K1()
				dgout = 1/Zout * ddZout - gout**2
			
				if np.isinf(gout):
					gout = 0
				if np.isinf(dgout):
					dgout = 0

		elif self.K == 2 :
			Zout = self.Zout_K2()
			if Zout == 0 or Zout < self.threshold_zero:
				gout = np.zeros((1,self.K))
				dgout = np.zeros((self.K,self.K))
			else :
				dZout = self.dZout_K2()
				gout = 1/Zout * dZout
				ddZout = self.ddZout_K2()
				gout = gout.reshape(self.K,1)
				dgout = 1/Zout * ddZout - gout.dot(gout.transpose())
				gout = gout.reshape(self.K)

		if self.print_Chrono_global : 
			end = time.time()
			print('Time gout_dgout=',end-start)

		return (gout , dgout)
	def dfout(self):
		norm = sqrt(det(self.V_inv)) /  ( (2*pi)**(self.K/2) )
		if self.K == 2 :
			dZout = self.dZout_K2()
		elif self.K == 1 :
			dZout = self.dZout_K1()
		dfout = norm * dZout
		return dfout
	def ddfout(self):
		norm = sqrt(det(self.V_inv)) /  ( (2*pi)**(self.K/2) )
		if self.K == 2 :
			ddZout = self.ddZout_K2()
		elif self.K == 1 :
			ddZout = self.ddZout_K1()
		ddfout = norm * ddZout
		return ddfout
	def fout(self):
		norm = sqrt(det(self.V_inv)) /  ( (2*pi)**(self.K/2) )		

		# K == 2
		if self.K == 2 :
			Zout = self.Zout_K2()
			if Zout == 0 or Zout < self.threshold_zero :
				Zout = 0

		# K == 1 
		elif self.K == 1 : 
			Zout = self.Zout_K1()
			if Zout == 0 or Zout < self.threshold_zero :
				Zout = 0

		fout =  norm * Zout

		return (fout)



