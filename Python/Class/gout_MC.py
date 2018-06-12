import paths
from libraries import *

class gout_MC(object):
	def __init__(self,K=1,y=0,omega=0,V=0,V_inv=0):
		self.K = K
		
		self.y = y 
		self.omega = (omega).reshape(1,self.K)[0]
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

		self.N_iter_MCquad = 10000
		self.NProcs = 1

		self.W1 = self.W1 = np.ones((self.K))
		self.sigma = 0.001


	# MC
	def sampler_mcquad(self,size):
		#print(omega)
		z =  (multivariate_normal.rvs(self.omega, self.V , size))
		return z 
	def integrand_gout_mcquad(self,Z):
		z = Z
		y = self.y
		omega = self.omega
		Pout = self.Pout(y,z)

		norm = np.array([1])
		avg = (z - omega)
		u = avg.reshape(self.K,1)
		var = u.dot(u.transpose())
		shape_var = var.shape
		var = var.reshape((1,self.K**2))[0]
		term = np.concatenate((norm,avg,var))

		integral = Pout *  term
		return integral
	def Phi_1(self,z1):
		resul = 1*(z1>0) -1*(z1<0) + 0*(z1==0)
		return resul
	def Phi_2(self,z2):
		resul = 1*(z2>0) -1*(z2<0) + 0*(z2==0)
		return resul
	def Pout(self,yl,z): # y scalar and z vector
		z = z.reshape(self.K,1)
		self.W1 =np.squeeze(np.asarray(self.W1))
		z1 = self.W1.transpose().dot(self.Phi_2(z))
		pout = np.exp( -1/(2*self.sigma**2)* ( yl - self.Phi_1(z1) )**2 )
		pout = pout.reshape(1)
		return pout

	def gout_dgout_mcquad(self):
		if self.print_Chrono_global :	
			start = time.time()

		result , error1 = mcimport(self.integrand_gout_mcquad, self.N_iter_MCquad , self.sampler_mcquad ,nprocs = self.NProcs )
	
		Zout = np.array(result[0])
		gout_avg = np.array(result[1:self.K+1])
		gout_var = np.array(result[self.K+1 : ]).reshape((self.K,self.K))
		if Zout == 0 or Zout<self.threshold_zero: 
			gout = np.zeros((self.K))
			dgout = np.zeros((self.K,self.K))
		else :
			gout = self.V_inv.dot(gout_avg)/Zout
			dgout = (self.V_inv.dot(gout_var).dot(self.V_inv))/Zout - self.V_inv - gout.reshape(self.K,1).dot(gout.reshape(1,self.K))

		if self.print_Chrono_global : 
			end = time.time()
			print('Time gout_dgout=',end-start)
		return (gout , dgout)


