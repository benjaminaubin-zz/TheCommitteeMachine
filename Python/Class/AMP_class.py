# gout
channel = 'sign-sign'
from gout_sign_sign import *
from Yhat_generror_K2 import *

################################## SAVE OBJECT ##################################
def save_object(obj,filename_object):
	with open(filename_object, 'wb') as output:
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
	print('Object saved:',filename_object)
def load_object(filename_object):
	with open(filename_object, 'rb') as input:
		obj = pickle.load(input)
	print("Object loaded:",filename_object)
	return obj

################################### AMP CLASS ###################################
class ApproximateMessagePassing(object):
	def __init__(self,K=1,PW_choice='binary',N=1000,alpha=1,MC_activated=False,seed=False,save_backup_running=True,print_Running=True):
		## Parameters
		self.K = K 
		self.N = N
		self.alpha = alpha
		self.P = int(self.N*self.alpha)
		self.channel = channel  #'sign-sign' or 'linear-relu'
		self.sigma = 0.001 # variance in the channel to repalce the Dirac delta

		## Seed 
		self.seed = seed
		if self.seed : 
			np.random.seed(0)
			print('SEED used for tests')

		## Monte Carlo 
		self.MC_activated = MC_activated
		if self.MC_activated : 
			print('Use MONTE CARLO to compute gout/dgout')

		## Weights
		self.W1 = np.ones((K)) # vector with only ones !
		self.W2 = np.zeros((K,N))
		self.PW_choice = PW_choice # 'binary' or 'gaussian'
		self.T_W = np.zeros((K)) # mean for gaussian
		self.Sigma_W = np.identity(K) # covariance for gaussian

		## Path
		self.path_Fig = 'Figures/'
		if not self.MC_activated :
			self.path_BackupObj = 'Data/BackupObj_AMP_'+ PW_choice +'_K='+ str(K)+'/' 
		else : 
			self.path_BackupObj = 'Data/BackupObj_AMP_MC_'+ PW_choice +'_K='+ str(K)+'/' 
		#if not os.path.exists(self.path_BackupObj):
		#	os.makedirs(self.path_BackupObj) 

		self.save_backup_running = save_backup_running
		self.file_name = self.path_BackupObj  + 'AMP_obj_tmp_'+PW_choice+'_K='+str(K)+'_alpha='+'%.4f'% alpha+'.pkl'
		
		## Options
		self.plot_Fig = True
		self.save_Fig = False
		
		## Option prints
		self.print_Chrono = False
		self.print_Initialization = False
		self.print_Update = False
		self.print_gout = False
		self.print_Running = print_Running
		self.print_Errors = False

		## Use_Tensor = True: much faster
		self.use_Tensor = True

		## Integration 
		self.borne_inf = -10
		self.borne_sup = +10

		## Convergence of the algorithm
		self.precision_q = 1e-5
		self.N_step_AMP = 100
		self.precision_What = 1e-2

		## Thresholding 
		self.threshold_zero = 1e-8
		self.threshold_inf = 1e+3
		self.threshold_q_stop = 0.995

		## Damping
		self.damping_activated = False
		self.damping_coef = 0.1

		## Data: gaussian with zero mean and variance 1/N
		self.T_X = 0 # zero mean 
		self.Sigma_X = 1/sqrt(self.N) # standard deviation 1/sqrt(N)
		self.X = np.zeros((self.N,self.P))
		self.Y = np.zeros(self.P)

		## Initialization near the ground truth solution 
		self.initialization_Truth = False

		### Storage
		# Messages
		self.W_hat = np.zeros((self.K,self.N))
		self.C_hat = np.zeros((self.K,self.K,self.N))
		# Variance and Mean : V and omega
		self.V = np.zeros((self.K,self.K,self.P))
		self.V_inv = np.zeros((self.K,self.K,self.P))
		self.omega = np.zeros((self.K,self.P))
		# gout dgout
		self.gout = np.zeros((self.K,self.P))
		self.dgout = np.zeros((self.K,self.K,self.P))
		# Sigma, Simga_inv, T
		self.Sigma = np.zeros((self.K,self.K,self.N))
		self.Sigma_inv = np.zeros((self.K,self.K,self.N))
		self.T = np.zeros((self.K,self.N))

		### Evolution optimization
		self.evolution_overlap = []
		self.q = np.zeros((self.K,self.K))

	############## Generate data ##################
	def generate_W2_W1(self):
		### Generate W2^*, W1^*
		# gaussian 
		if self.PW_choice == 'gaussian':
			for i in range(self.N):
				self.W2[:,i] = np.random.multivariate_normal(self.T_W, self.Sigma_W )		
			self.W1 = np.ones((1,self.K))
		# binary 
		elif self.PW_choice == 'binary':
			self.W2 = 2 * np.random.randint(2, size=(self.K, self.N)) - 1
			self.W1 = np.ones((1,self.K))
		else : 
			print('Error: Prior PW not defined')

		if self.print_Initialization:
			print('W2=',self.W2,'\n')
			print('W1=',self.W1,'\n')

		return (self.W2 , self.W1)
	def generate_X(self):
		### Generate X
		self.X = np.random.normal(self.T_X, self.Sigma_X , (self.N,self.P))
		if self.print_Initialization:
			print('X=',self.X,'\n')
		return self.X
	# Pout for sign-sign
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
	def phi_out(self,X_new):
		z = self.W2.dot(X_new)
		z = z.reshape(self.K,1)
		self.W1 =np.squeeze(np.asarray(self.W1))
		z1 = self.W1.transpose().dot(self.Phi_2(z))
		phi_out = self.Phi_1(z1) 
		return phi_out
	def generate_Y(self):
		Z = self.W2.dot(self.X)
		#print('Z=',Z)
		Y2 = self.Phi_2(Z)
		#print('Y2=',Y2)
		Z1 = (self.W1).dot(Y2)
		#print('Z1=',Z1)
		Y = (self.Phi_1((Z1).transpose())).reshape(self.P)
		self.Y = Y
		#print('Y=',Y)
		if self.print_Initialization:
			print('Y=',self.Y,'\n')
		return (self.Y)
	
	############## Initialisation ##################
	# Global intitialization
	def initialization(self):
		print('Start initialization')
		if self.print_Initialization :
			print('################# INITIALIZATION #################','\n')
			print('### Generate X, Y, W2, W1 ###')
		mode = 1
		if mode == 1 :
			self.generate_W2_W1()
			self.generate_X()
			self.generate_Y()
		
			self.initialization_What_Chat()
			self.initialization_V_omega()
			self.check_V_is_definite_positive()

			self.initialization_gout_dgout() 

			self.initialization_Sigma_T()

		if self.print_Initialization:
			print('############ INITIALIZATION COMPLETED ############','\n')
		print('Successful initialization','\n')
	# Single intitialization
	def initialization_What_Chat(self):
		W_hat = np.zeros((self.K,self.N))
		C_hat = np.zeros((self.K,self.K,self.N))
		for i in range(self.N):
			if self.initialization_Truth : 
				noise = 0.1
				W_hat[:,i] = self.W2[:,i] + noise * np.random.randn(self.K)
				C_hat[:,:,i] = 0.5 * np.identity(self.K) 
				#C_hat[:,:,i] = self.initialization_Chat() 
			else :
				W_hat[:,i] = self.initialization_What()
				C_hat[:,:,i] = self.initialization_Chat() 
		self.W_hat = W_hat
		self.C_hat = C_hat
		if self.print_Initialization:
			print('### What, Chat Initialized ###')
			self.print_matrix(self.W_hat,'What =')
			self.print_matrix(self.C_hat,'Chat =')
		return (self.W_hat,self.C_hat)
	def initialization_What(self):
		W_hat = np.zeros((self.K))
		if self.PW_choice == 'gaussian':
			W_hat = self.T_W 
		if self.PW_choice == 'binary':
			W_hat = np.zeros((self.K))
		return W_hat
	def initialization_Chat(self):
		C_hat = np.zeros((self.K,self.K))
		if self.PW_choice == 'gaussian':
			C_hat = self.Sigma_W 
		if self.PW_choice == 'binary':
			C_hat = np.identity(self.K)
		return C_hat
	def initialization_V_omega(self):
		V = np.zeros((self.K,self.K,self.P))
		V_inv = np.zeros((self.K,self.K,self.P))
		mode_V = 4 
		for l in range(self.P):
			if mode_V == 1 : # Symmetric diagonal matrix 
				V_tmp = np.random.randn(self.K,self.K)
				#V[:,:,l] = 1/10*(V_tmp.dot(V_tmp)) + np.identity(self.K) # to  be symmetric
				V[:,:,l] = (V_tmp + V_tmp.transpose() )/2 +self.K*np.eye(self.K) # to  be symmetric
			if mode_V == 2 : # Wishart 
				V[:,:,l] = wishart.rvs(df, scale, size=(self.K,self.K), random_state=None)
			if mode_V == 3 : # Identity
				V[:,:,l] = self.K*np.identity(self.K)
			if mode_V == 4 : # SPD
				V[:,:,l] = make_spd_matrix(self.K)
			if mode_V == 5 : 
				qd , qa = 0.1 , 0.001 
				V[:,:,l] = np.identity((self.K)) - ( (qd) * np.identity(self.K) + qa/self.K * np.ones((self.K,self.K)) )
			
			V_inv[:,:,l] = inv(V[:,:,l])

		mode_omega = 2
		if mode_omega == 1 :
			omega = np.ones((self.K,self.P))
		if mode_omega == 2 : 
			omega = np.random.randn(self.K,self.P)

		self.V = V
		self.V_inv = V_inv
		self.omega = omega

		if self.print_Initialization:
			print('### V, V_inv, omega Initialized ###')
			self.print_matrix(self.V,'V =')
			self.print_matrix(self.V_inv,'V_inv =')
			self.print_matrix(self.omega,'omega =')
			l = np.random.randint(0,self.P)
			print('### Check that V is PD:',self.check_is_definite_positive(self.V[:,:,l]),'\n')
		return (self.V ,self.V_inv, self.omega)
	def initialization_V_omega_modif(self):
		V = np.zeros((self.K,self.K,self.P))
		V_inv = np.zeros((self.K,self.K,self.P))
		omega = np.zeros((self.K,self.P))
		for l in range(self.P):
			V[:,:,l] = self.compute_V(l)
			V_inv[:,:,l] = inv(V[:,:,l])
			omega[:,l] = self.compute_omega_initial(l)
		self.V = V
		self.V_inv = V_inv
		self.omega = omega
		return (self.V ,self.V_inv, self.omega)
	def initialization_gout_dgout(self):
		(self.gout,self.dgout) = self.update_gout_dgout()
		if self.print_Initialization : 
			print('### gout, dgout Computed ###')
			self.print_matrix(self.gout,'gout =')
			self.print_matrix(self.dgout,'dgout =')
			print('\n')
	def initialization_Sigma_T(self):
		if self.use_Tensor:
			self.Sigma_inv = self.update_Sigma_inv_tensor()
			self.Sigma = self.update_Sigma()
			self.T = self.update_T_tensor()
		else :
			self.Sigma_inv= self.update_Sigma_inv()
			self.Sigma= self.update_Sigma()
			self.T = self.update_T()
		if self.print_Initialization:
			print('### S, T Computed ###')
			self.print_matrix(self.Sigma,'Sigma =')
			self.print_matrix(self.Sigma_inv,'Sigma_inv =')
			self.print_matrix(self.T,'T =')
		return (self.Sigma , self.Sigma_inv , self.T) 

	def initialization_test(self):
		if self.print_Initialization :
			print('################# INITIALIZATION #################','\n')
			print('### Generate X, Y, W2, W1 ###')

		self.generate_W2_W1()
		self.generate_X()
		self.generate_Y()
			
		self.initialization_What_Chat()
		self.initialization_V_omega()
		#self.initialization_V_omega_modif() # Seems to work better ! 

		if self.print_Initialization:
			print('############ INITIALIZATION COMPLETED ############','\n')
	############## Update ##############
	def damping(self,X_new,X_self):
		alpha = self.damping_coef
		return  (1-alpha) * (X_self) + ( alpha ) * X_new

	# Sigma, Sigma_inv (K,K,N)
	def update_Sigma_inv_tensor(self):
		Sigma_inv = - np.einsum('ijl,kl->ijk',self.dgout,np.square(self.X))
		return (Sigma_inv)
	def update_Sigma_inv(self):
		Sigma_inv = np.zeros((self.K,self.K,self.N))
		for i in range(self.N):
			Sigma_tmp_inv = self.compute_Sigma_inv(i)
			Sigma_inv[:,:,i] = Sigma_tmp_inv
		return (Sigma_inv)
	def compute_Sigma_inv(self,i): 
		Sigma_inv_tmp = np.zeros((self.K,self.K))
		for l in range(self.P):
			Sigma_inv_tmp +=  -(self.X[i,l])**2 * ( self.dgout[:,:,l] )
		return Sigma_inv_tmp
	def update_Sigma_tensor(self):
		Sigma = np.moveaxis(inv(np.moveaxis(self.Sigma_inv, -1, 0)), 0, -1)
		return Sigma
	def update_Sigma(self):
		Sigma = np.zeros((self.K,self.K,self.N))
		for i in range(self.N):
			Sigma[:,:,i] = inv(self.Sigma_inv[:,:,i])
		return Sigma
	# T (K,N)
	def update_T_tensor(self):
		tmp_1 = np.einsum('ij,lj->il',self.gout,self.X)
		term1 = np.einsum('ijl,jl->il',self.Sigma,tmp_1)

		tmp_2 = - np.einsum('ijk,lk->ijl',self.dgout,np.square(self.X))
		tmp_3 = np.einsum('ijk,jk->ik',tmp_2,self.W_hat)
		term2 = np.einsum('ijk,jk->ik',self.Sigma,tmp_3)
		T = term1 + term2
		return T
	def update_T(self):
		T = np.zeros((self.K,self.N))
		for i in range(self.N):
			T[:,i] = self.compute_T(i)
		return T
	def compute_T(self,i):
		T_tmp = np.zeros((self.K))
		for l in range(self.P):
			T_tmp += self.X[i,l] * self.gout[:,l] - (self.X[i,l])**2 * (self.dgout[:,:,l]).dot(self.W_hat[:,i])
		#T_tmp = np.sum([self.X[i,l] * self.gout[:,l] - (self.X[i,l])**2 * (self.dgout[:,:,l]).dot(self.W_hat[:,i]) for  l in range(self.P)])
		T_tmp =  (self.Sigma[:,:,i]).dot(T_tmp)
		return T_tmp
	
	# Compute gout, dgout
	def update_gout_dgout(self):
		gout = np.zeros((self.K,self.P))
		dgout = np.zeros((self.K,self.K,self.P))
		for l in range(self.P):
			if self.print_gout:  
				print('gout =',l/self.P*100.,'%')
			gout[:,l] , dgout[:,:,l] = self.compute_gout_dgout(l)		
		return (gout , dgout)
	def compute_gout_dgout(self,l):
		# Sign-Linear
		if self.K > 0 and self.channel == 'sign-linear': 
			(gout,dgout) = self.gout_dgout_linear(l)
		
		# Sign-Sign for K=1
		elif self.channel == 'sign-sign' :
			V = self.V[:,:,l]
			V_inv = self.V_inv[:,:,l]
			omega = self.omega[:,l]
			y = self.Y[l]
			
			# Sign-Sign for K = 1, K=2
			if (self.K ==1 or self.K==2 ) and not self.MC_activated:  
				gout_ss = gout_sign_sign(K=self.K,y=y,omega=omega,V=V,V_inv=V_inv)
				(gout, dgout) = gout_ss.gout_dgout()
			# Sign-Sign for larger K
			else :
				## Sum config
				#gout_ss = gout_sign_sign_sum_config(K=self.K,y=y,omega=omega,V=V,V_inv=V_inv)
				#(gout, dgout) = gout_ss.gout_dgout()
				#print('NEED TO CORRECT gout_sign_sign_sum_config - wrong results')
				
				## Monte Carlo
				gout_mc = gout_MC(K=self.K,y=y,omega=omega,V=V,V_inv=V_inv)
				(gout,dgout) =  gout_mc.gout_dgout_mcquad()

				## Using expansion at large K
				#gout_ss = gout_sign_sign_largeK(K=self.K,y=y,omega=omega,V=V)
				#(gout, dgout) = gout_ss.gout_dgout()

		elif self.channel == 'linear-relu':
			V = self.V[:,:,l]
			V_inv = self.V_inv[:,:,l]
			omega = self.omega[:,l]
			y = self.Y[l]
			gout_lr = gout_linear_relu(K=self.K,y=y,omega=omega,V=V,V_inv=V_inv)
			(gout , dgout) = gout_lr.gout_dgout()

		else : 
			print('No expression for gout')
			#self.print_Running = True
			#(gout,dgout) =  self.gout_dgout_mcquad(l)

		return (gout , dgout)

	# What (K,N) , Chat (K,K,N)
	def update_What_Chat(self): # Update messages
		W_hat = np.zeros((self.K,self.N))
		C_hat = np.zeros((self.K,self.K,self.N))
		for i in range(self.N):
			W_hat[:,i] , C_hat[:,:,i]= self.compute_fW_fC(i)

		return (W_hat,C_hat)
	def compute_fW_fC(self,i): 
		Sigma , Sigma_inv , T = self.Sigma[:,:,i] , self.Sigma_inv[:,:,i] , self.T[:,i]
		( fW , fC ) = np.zeros((self.K)) , np.zeros((self.K,self.K))
		# Gaussian case
		if self.PW_choice == 'gaussian' :
			Sigma_star = inv(inv(self.Sigma_W)+Sigma_inv)
			T_star = Sigma_star.dot( inv(self.Sigma_W).dot(self.T_W) + Sigma_inv.dot(T) )
			fW = T_star  #fW = T - Sigma.dot( inv(self.Sigma_W + Sigma) ).dot(T-self.T_W)
			fC = Sigma_star #fC = Sigma.dot(np.identity(self.K) - Sigma.dot( inv(self.Sigma_W + Sigma) ) )

		# Binary for K =1
		elif self.PW_choice == 'binary' and self.K == 1:
			arg_neg = 	-1/2 * (1-T)**2 * Sigma_inv 
			arg_pos = 	-1/2 * (1+T)**2 * Sigma_inv 
			#print(arg_neg,arg_pos,Sigma_inv,T)

			Z = 1/2 * ( exp(arg_neg) + exp(arg_pos))
			if Z == 0 :
				fW = 0
			else : 
				fW = 1/2 * ( exp(-1/2 * (1-T)**2 * Sigma_inv ) - exp(-1/2 * (1+T)**2 * Sigma_inv )) / Z
				fC = 1 - fW**2

		# Binary for K >1 
		elif self.PW_choice == 'binary' and self.K > 1:
			( fW , fC ) = self.integrals_fW_fC(i)
			#fW = fW.reshape(self.K)
			
		else :
			print('Prior PW not defined')
		return ( fW , fC ) 
	
	def configuration_W(self,n):
		W = np.array( [int(x) for x in list('{0:0b}'.format(n).zfill(self.K)) ] )
		W = 2 * W - 1
		return W
	def integrals_fW_fC(self,i):
		Sigma , Sigma_inv , T = self.Sigma[:,:,i] , self.Sigma_inv[:,:,i] , self.T[:,i]
		(Z, W_avg , W_squared_avg ) = 0 , np.zeros((self.K)) , np.zeros((self.K,self.K))

		for i in range(2**self.K) :
			W = self.configuration_W(i)
			#W = W.reshape(self.K,1)
			argument = - 1/2 * ((W-T).transpose()).dot(Sigma_inv).dot(W-T)			
			try :
				weight = exp(argument)
			except : 
				weight = self.threshold_zero 
			Z += weight
			W_avg += weight * W
			W_squared_avg += weight * W.reshape(self.K,1).dot(W.reshape(self.K,1).transpose())

		Z /= (2**self.K)
		W_avg /= (2**self.K)
		W_squared_avg /= (2**self.K)

		if np.isnan(Z) or Z == 0 or not np.isfinite(Z):
			if self.print_Errors:
				print('Error: fWC_binary_K Z:',Z)
			Z = 0
			f_W = np.zeros((self.K))
			f_C = np.zeros((self.K,self.K))

		if Z == 0 or Z < self.threshold_zero : 
			(Z, fW , fC ) = 0 , np.zeros((self.K)) , np.zeros((self.K,self.K))
		else : 
			fW = W_avg/Z
			fC = W_squared_avg/Z - fW.reshape(self.K,1).dot(fW.reshape(self.K,1).transpose())
		return ( fW , fC )

	# V, Vinv (K,K,P)
	def update_V_tensor(self):
		V = np.einsum('ijl,lk->ijk',self.C_hat,np.square(self.X)) 
		#V_inv = np.zeros((self.K,self.K,self.P))
		#for l in range(self.P):
		#	V_inv[:,:,l] = inv(V[:,:,l])
		return (V)
	def update_V_inv_tensor(self):
		V_inv = np.moveaxis(np.linalg.inv(np.moveaxis(self.V, -1, 0)), 0, -1)
		return V_inv
	def update_V_inv(self):
		V_inv = np.zeros((self.K,self.K,self.P))
		for l in range(self.P):
			V_inv[:,:,l] = inv(self.V[:,:,l])
		return V_inv
	def update_V(self):
		V = np.zeros((self.K,self.K,self.P))
		V_inv = np.zeros((self.K,self.K,self.P))
		for l in range(self.P):
			V[:,:,l] = self.compute_V(l)
			V_inv[:,:,l] = inv(V[:,:,l])
		return (V)
	def compute_V(self,l): 
		V_tmp = np.zeros((self.K,self.K))
		for i in range(self.N):
			V_tmp += (self.X[i,l])**2 * self.C_hat[:,:,i]
		#V_tmp = np.sum([(self.X[i,l])**2 * self.C_hat[:,:,i] for i in range(self.N) ],axis=0)
		return V_tmp

	def V_committee_sym(self,V_tensor_old):
		V_tensor = np.zeros((self.K,self.K,self.P))
		for i in range(self.P):
			V = V_tensor_old[:,:,i]
			diag = np.diag(V)
			off_diag = (np.sum(V) - np.sum(diag) )/(self.K**2 - self.K)
			V_new = np.ones((self.K,self.K)) * off_diag - np.diag(np.ones((self.K))*off_diag) + np.diag(diag)
			V_tensor[:,:,i] = V_new 
		return V_tensor

	# Omega (K,P)
	def update_omega_tensor(self):
		#ATTENTION : SIGMA, SIMGA_Inv MISSING, but seems to commute!  
		term1 = np.einsum('ij,jl->il',self.W_hat,self.X)
		#term_C_Sigma = np.einsum('ijk,jik->ijk',self.C_hat,self.Sigma)
		#term_Sigma_inv_C_Sigma = np.einsum('ijk,jik->ijk',self.Sigma_inv,term_C_Sigma)
		term_Sigma_inv_C_Sigma = self.C_hat
		tmp = np.einsum('ijk,jl->ikl',term_Sigma_inv_C_Sigma,self.gout)
		term2 = np.einsum('ijl,jl->il',tmp,np.square(self.X))
		omega = term1 - term2
		return omega
	def update_omega(self):
		omega = np.zeros((self.K,self.P))
		for l in range(self.P):
			omega[:,l] = self.compute_omega(l)
		return omega 
	def compute_omega(self,l): # Omega of size (K,P)
		omega_tmp = np.zeros((self.K))
		for i in range(self.N):
			omega_tmp += self.X[i,l] * self.W_hat[:,i] - self.X[i,l]**2 * ((self.Sigma_inv[:,:,i].dot(self.C_hat[:,:,i])).dot(self.Sigma[:,:,i])).dot(self.gout[:,l])
			#omega_tmp += self.X[i,l] * self.W_hat[:,i] - self.X[i,l]**2 * self.C_hat[:,:,i].dot(self.gout[:,l])
		#omega_tmp = np.sum([self.X[i,l] * self.W_hat[:,i] - self.X[i,l]**2 * ((self.Sigma_inv[:,:,i].dot(self.C_hat[:,:,i])).dot(self.Sigma[:,:,i])).dot(self.gout[:,l]) for i in range(self.N) ],axis=0)
		#omega_tmp = np.sum([self.X[i,l] * self.W_hat[:,i] - self.X[i,l]**2 * (self.C_hat[:,:,i]).dot(self.gout[:,l]) for i in range(self.N) ])
		return omega_tmp
	def compute_omega_initial(self,l):
		omega_tmp = np.zeros((self.K))
		for i in range(self.N):
			omega_tmp += self.X[i,l] * self.W_hat[:,i]
		return omega_tmp

	############## Step ##############
	def AMP_step_t(self,t): 
		start_0 = time.time()
		if self.print_Update : 
			print('################# START UPDATE TIME',t,'#################','\n')
		
		#### Update of What, Chat
		# Chrono
		if self.print_Chrono:
			start = time.time()
		# Update
		(W_hat_t,C_hat_t) = self.update_What_Chat()
		# Chrono
		if self.print_Chrono:
			t_tmp_1 = time.time()
			print('update_What_Chat:',t_tmp_1-start)
		# Damping
		if self.damping_activated : 
			self.W_hat = self.damping(W_hat_t,copy.copy(self.W_hat) )
			self.C_hat = self.damping(C_hat_t,copy.copy(self.C_hat) )
		else :
			self.W_hat = W_hat_t
			self.C_hat = C_hat_t
		# Print
		if self.print_Update:
			print('### What, Chat at time',t,'###','\n')
			self.print_matrix(self.W_hat,'What =')
			self.print_matrix(self.C_hat,'Chat =')

		#### Update of V and V_inv
		# Chrono
		if self.print_Chrono:
			start = time.time()
		# Update
		if self.use_Tensor:
			(V_t) = self.update_V_tensor()
		else :
			(V_t) = self.update_V()
			#self.print_matrix(V_t,'V_t=')
			(V_t_tensor) = self.update_V_tensor()
			#self.print_matrix(V_t,'V_t=')
			print('difference V',np.allclose(V_t_tensor,V_t))

		#print('V=',V_t[:,:,0])


		# Chrono
		if self.print_Chrono:
			t_tmp = time.time()
			print('update_V:',t_tmp-start)
		# Damping
		if self.damping_activated :
			self.V = self.damping(V_t,copy.copy(self.V) )
		else :
			self.V = V_t

		# ATTENTION : NO DAMPING for the inverse. Danmping AND THEN DAMPING!
		if self.use_Tensor:
			self.V_inv = self.update_V_inv_tensor()
		else : 
			self.V_inv = self.update_V_inv()

		#self.check_V_is_definite_positive()

		#### Update of omega
		# Chrono
		if self.print_Chrono:
			start = time.time()
		# Update
		if self.use_Tensor:
			(omega_t) = self.update_omega_tensor()
		else :
			(omega_t) = self.update_omega()
			#self.print_matrix(omega_t,'omega_t=')
			(omega_t_tensor) = self.update_omega_tensor()
			#self.print_matrix(omega_t_tensor,'omega_t=')
			print('difference omega',np.allclose(omega_t_tensor,omega_t))
		# Chrono
		if self.print_Chrono:
			t_tmp = time.time()
			print('update_omega:',t_tmp-start)
		# Damping
		if self.damping_activated :
			self.omega = self.damping(omega_t,copy.copy(self.omega) )
		else : 
			self.omega = omega_t
		# Print
		if self.print_Update:
			print('### V, V_inv, omega at time',t,'###','\n')
			self.print_matrix(self.V,'V =')
			self.print_matrix(self.V_inv,'V_inv =')
			self.print_matrix(self.omega,'omega =')

		#### Update gout, dgout
		# Chrono
		if self.print_Chrono:
			start = time.time()	
		# Update
		(gout_t,dgout_t) = self.update_gout_dgout()
		# Chrono
		if self.print_Chrono:
			t_tmp = time.time()
			print('update_gout:',t_tmp-start)
		# Damping
		if self.damping_activated :
			self.gout = self.damping(gout_t,copy.copy(self.gout) ) 
			self.dgout = self.damping(dgout_t,copy.copy(self.dgout) )
		else : 
			self.gout = gout_t
			self.dgout = dgout_t
		# Print
		if self.print_Update or self.print_gout:
			print('### gout, dgout at time',t,'###','\n')
			self.print_matrix(self.gout,'gout =')
			self.print_matrix(self.dgout,'dgout =')

		#### Update Sigma, Sigma_inv
		# Chrono
		if self.print_Chrono:
			start = time.time()
		# Update
		if self.use_Tensor:
			(Sigma_inv_t) = self.update_Sigma_inv_tensor()
		else :
			(Sigma_inv_t) = self.update_Sigma_inv()
			#self.print_matrix(Sigma_inv_t,'Sigma_inv_t=') 
			(Sigma_inv_t_tensor ) = self.update_Sigma_inv_tensor()
			#self.print_matrix(Sigma_inv_t_tensor,'Sigma_inv_t_tensor=') 
			print('difference Sigma',np.allclose(Sigma_inv_t_tensor,Sigma_inv_t))

		# Chrono
		if self.print_Chrono:
			t_tmp = time.time()
			print('update_Sigma:',t_tmp-start)
		# Damping
		if self.damping_activated :
			self.Sigma_inv = self.damping(Sigma_inv_t,copy.copy(self.Sigma_inv) )
		else : 
			self.Sigma_inv = Sigma_inv_t
		# ATTENTION : NO DAMPING for the inverse. Danmping AND THEN DAMPING!
		if self.use_Tensor:
			self.Sigma = self.update_Sigma()
		else : 
			self.Sigma = self.update_Sigma()

		#### Update T
		# Chrono
		if self.print_Chrono:
			start = time.time()
		# Update
		if self.use_Tensor:
			T_t = self.update_T_tensor()
		else :
			T_t = self.update_T()
			#self.print_matrix(T_t,'T_t=') 
			T_t_tensor = self.update_T_tensor()
			#self.print_matrix(T_t_tensor,'T_t=') 
			print('difference T',np.allclose(T_t_tensor,T_t))
		# Chrono
		if self.print_Chrono:
			t_tmp = time.time()
			print('update_T:',t_tmp-start)
		# Damping
		if self.damping_activated :
			self.T = self.damping(T_t,copy.copy(self.T) )
		else : 
			self.T = T_t
		# Print
		if self.print_Update:
			print('### Sigma, Sigma_inv, T at time',t,'###','\n')
			self.print_matrix(self.Sigma,'Sigma =')
			self.print_matrix(self.Sigma_inv,'Sigma_inv =')
			self.print_matrix(self.T,'T =')

		if self.print_Update:
			print('################# END UPDATE TIME',t,'#################')
			print('\n')

		end = time.time() 
		#if self.print_Running :
			#print('Time ellapsed for the step =',end - start_0)

		return (self.W_hat,self.C_hat,self.V,self.V_inv,self.omega,self.Sigma,self.Sigma_inv,self.T,self.gout,self.dgout)
	def AMP_iteration(self):
		print('K=',self.K,'PW=',self.PW_choice,'alpha=',self.alpha)
		print('Start AMP:',time.ctime())
		difference , step , stop = 10 , 0 , False
		self.start = time.ctime()
		q = self.overlap()

		AMP_storage_alpha = AMP_data_alpha(self.K,self.PW_choice,self.N,1)
		if self.save_backup_running :
			save_object(AMP_storage_alpha,self.file_name)	 

		while step < self.N_step_AMP  and stop == False and difference > self.precision_q:
			step_ = step
			#if self.print_Running:
			#	print('Step = ',step)
			step += 1

			q = copy.copy(self.evolution_overlap[-1]) 
			W_hat = copy.copy(self.W_hat) 

			if np.amax(self.q) > 0.99 : 
				try :
					self.AMP_step_t(step)
				except : 
					break
			else :
				self.AMP_step_t(step)

			#if self.print_Running : 
			#	self.print_matrix(self.q,'q_AMP=')

			# Compute difference for convergence
			self.overlap()
			difference_q = norm(self.q-q)
			difference_What = norm(self.W_hat-W_hat)
			if self.print_Running:
				if step_ == 1 or step_ % 5 == 0 :
					print('Step =',step_,'Diff_q =',difference_q,'Diff_What =',difference_What)
				#print('difference q =',difference_q)
				#print('difference What =',difference_What)
			difference = difference_q
			self.difference = difference

			AMP_storage_alpha.compute_averaged_q([self])
			AMP_storage_alpha.step += 1
			if self.save_backup_running :
				save_object(AMP_storage_alpha,self.file_name)
			
			if self.PW_choice == 'binary':
				if np.amax(self.q) > self.threshold_q_stop:
					stop = True
			if self.PW_choice == 'gaussian':
				if np.amax(self.q) > self.threshold_q_stop and step > 25 :
					stop = True

		self.print_matrix(self.q,'Final overlap: q_AMP=')
		print('End AMP')

	def overlap(self):
		What = self.W_hat
		q = 1/self.N * What.dot(What.transpose())
		(self.evolution_overlap).append(q)
		self.q = q
		#if self.print_Running:
			#self.print_matrix(self.q,'q_AMP=')
		return q
	############## Annex functions ##############
	def check_is_definite_positive(self,M):
		is_def = np.all(np.linalg.eigvals(M) > 0)
		return is_def
	def check_is_definite_negative(self,M):
		is_def = np.all(np.linalg.eigvals(M) < 0)
		return is_def
	def print_matrix(self,M,txt):
		print(txt)
		try :
			n,m = M.shape
		except:
			n = len(M)
		for i in range(n):
			print(M[i,:])
	
	def check_V_is_definite_positive(self):
		not_def = []
		for l in range(self.P):
			test_V = self.check_is_definite_positive(self.V[:,:,l])
			test_V_inv = self.check_is_definite_positive(self.V_inv[:,:,l])
			det_V = det(self.V[:,:,l])
			det_Vinv = det(self.V_inv[:,:,l])
			if not test_V  or not test_V_inv :
				not_def.append([l,det_V,det_Vinv])
		if len(not_def) > 0 : 
			print('V IS NOT SEMI DEFINITE POSITIVE')
		#else : 
		#	print('V is DEFINITE POSITIVE')

	############## Generalization error ##############
	# Compute prediction for a new sample
	def Y_hat_func(self,X_new):
		mu = np.zeros((self.K,1))
		Sigma = np.zeros((self.K,self.K))
		for i in range(self.N):
			mu += X_new[i] * self.W_hat[:,i].reshape((self.K,1))
			Sigma += X_new[i]**2 *  ( self.C_hat[:,:,i] )
		
		V = Sigma
		Sigma_inv = inv(V)
		Yhat_generror = Yhat_generror_K2(K=self.K,omega=mu,V_inv=Sigma_inv)
		Y_hat = Yhat_generror.Yhat() 
		return Y_hat
	# Compute the generalization error for a test set of size N_samples
	def gen_error(self,N_samples):
		tab_gen = []
		print('Start Gen Error')
		gen_error = 0 
		for i in range(N_samples):
			if (i*100.0/N_samples % 10) == 0 :   
				print(i*100.0/N_samples , '%')
			X_new = np.random.normal(0, 1/sqrt(self.N), (self.N,1))
			Y_hat = self.Y_hat_func(X_new)
			Y = self.phi_out(X_new)
			#print('#',i,'Y=',Y,'Yhat=',Y_hat)
			gen_error += (Y-Y_hat)**2
			tab_gen.append(0.5*gen_error / (i+1))
		gen_error *= 0.5 / N_samples
		print('End Gen Error')
		print('Generalization Error =',gen_error[0])
		return gen_error,tab_gen
	############## Plots ##############
	def plot_q(self,obj_SE):
		title = r'$q_{AMP}(t)$ vs $q_{SE}(t)$ at $\alpha=$'+str(self.alpha)+' K='+str(self.K)
		Fontsize = 25
		Fontsize_ticks = 20

		fig, ax1 = plt.subplots(figsize=[8,8])
		colors = np.array([ [dc, cr ] ,[db , do ]   ])

		n = len(self.evolution_overlap);
		tab_q = np.array(self.evolution_overlap);
		tab_t = np.arange(0,n,1);

		for i in range(self.K):
			for j in range(self.K):
				data = tab_q[:,i,j];
				if (i == 1 and j == 1) or (i == 1 and j == 0) : 
					ax1.plot(tab_t,data,'-',color=colors[i,j],Markersize =3, Linewidth=1.5,label=r'AMP - $q$['+str(i)+','+str(j)+']')
				else : 
					ax1.plot(tab_t,data,'-',color=colors[i,j],Markersize =3, Linewidth=1.5)

		ax1.plot([min(tab_t),max(tab_t)],[obj_SE.q[0,1],obj_SE.q[0,1]],'--',color=colors[1,0],label=r'SE - $q$['+str(1)+','+str(0)+']')
		ax1.plot([min(tab_t),max(tab_t)],[obj_SE.q[0,0],obj_SE.q[0,0]],'--',color=colors[1,1],label=r'SE - $q$['+str(1)+','+str(1)+']')
		ax1.set_xlabel(r'time $t$',fontsize=Fontsize)
		ax1.set_ylabel(r'$q^t$',fontsize=Fontsize)
		plt.legend(loc='best', fontsize=17.5)
		plt.title(title,fontsize=Fontsize)
		ax1.set_ylim([0,1])
		ax1.set_xlim([0,max(tab_t)])
		ax1.tick_params(labelsize=Fontsize_ticks)
	def plot_gen_error(self,tab_gen_AMP,gen_SE):
		title = r'Generalization error AMP vs SE at $\alpha=$'+str(self.alpha)+' K='+str(self.K)
		Fontsize = 25
		Fontsize_ticks = 20

		fig, ax1 = plt.subplots(figsize=[8,8])
		colors = np.array([ [dc, cr ] ,[db , do ]   ])

		n = len(tab_gen_AMP);
		tab_t = np.arange(0,n,1);

		ax1.plot(tab_t,tab_gen_AMP,'--',color='k',Markersize =3, Linewidth=1.5,label=r'AMP - $\epsilon_g^t(\alpha)$')
		ax1.plot([min(tab_t),max(tab_t)],[gen_SE,gen_SE],'-',color='r',Markersize =3, Linewidth=1.5,label=r'SE - $\epsilon_g^t(\alpha)$')

		ax1.set_xlabel(r'$N_{samples}$',fontsize=Fontsize)
		ax1.set_ylabel(r'$\epsilon_g$',fontsize=Fontsize)
		plt.legend(loc='best', fontsize=17.5)
		plt.title(title,fontsize=Fontsize)
		ax1.set_ylim([min(tab_gen_AMP),max(tab_gen_AMP)*1.1])
		ax1.set_xlim([0,max(tab_t)])
		ax1.tick_params(labelsize=Fontsize_ticks)
	############## Backups ##############
	def get_all_backup(self,path_data):
		res = os.chdir(path_data)
		list_files = sorted(sorted([s for s in os.listdir() if s.endswith('.pkl')], key=os.path.getmtime))
		step = 0
		tab_alpha = np.zeros(len(list_files))

		for file in list_files:
			print(file)
			AMP_backup = load_object(file)
			alpha = AMP_backup.alpha
			tab_alpha[step] = alpha
			step +=1
		self.tab_alpha = tab_alpha
		self.tab_files = list_files
		os.chdir('../../')

################################### Storage ###################################
class AMP_data(object):
	def __init__(self,K,PW_choice,N,N_samples):
		self.K = K
		self.PW_choice = PW_choice 
		self.N = N
		self.N_samples = N_samples
		#self.list_all_obj = [] # list of list
		self.list_alpha = [] 
		self.list_averaged_q = []
		
		self.path_Figures = 'Figures/'
		self.plotFig = True
		self.saveFig = True

	def plot_results(self):
		title = 'AMP alpha K='+str(self.K)
		title_save = 'AMP_alpha_PW='+self.PW_choice+'_K'+str(self.K)+'.eps'

		db,do,g,cr,mv,o,dc,dg= 'dodgerblue','darkorange','gold','crimson','mediumvioletred','orangered','darkcyan','darkgreen'
		fig, ax1 = plt.subplots(figsize=[8,8])
		colors = np.array([ [dc, cr ] ,[db , do ]   ])


		list_alpha , list_averaged_q = zip(*sorted(zip(self.list_alpha , self.list_averaged_q)))
		tab_alpha = np.array(list_alpha)
		tab_q = np.array(list_averaged_q)
		#print(tab_q)

		for i in range(self.K):
			for j in range(self.K):
				data = tab_q[:,i,j]
				ax1.plot(tab_alpha,data,'.',color=colors[i,j],Markersize =2, Linewidth=1,label=r'$q$['+str(i)+','+str(j)+']')

		ax1.set_xlabel(r'$\alpha$',fontsize=Fontsize)
		ax1.set_ylabel(r'$q_{AMP}^t$',fontsize=Fontsize)
		plt.legend(loc='best', fontsize=Fontsize)
		plt.title(title,fontsize=Fontsize)
		ax1.set_ylim([0,1])
		ax1.set_xlim([0,max(tab_alpha)])

		if self.saveFig:
			if not os.path.exists(self.path_Figures):
				os.makedirs(self.path_Figures)
			plt.savefig(self.path_Figures+title_save, format='eps', dpi=1000,bbox_inches="tight")

		if self.plotFig:
			plt.show(block=False)
			input("Press Enter to continue...")
			plt.close()
		return fig
class AMP_data_alpha(object):
	def __init__(self,K,PW_choice,N,N_samples):
		self.K = K
		self.PW_choice = PW_choice 
		self.N = N
		self.N_samples = N_samples
		self.alpha = 0
		self.averaged_q = np.zeros((self.K,self.K))
		self.step = 0
		self.print_Running = False

	def compute_averaged_q(self,list_obj):
		averaged_q = np.zeros((self.K,self.K))
		ind = 0 
		for obj in list_obj:
			averaged_q += obj.q 
			if self.print_Running :
				print('#',ind,'q=',obj.q )
			ind +=1
		averaged_q /= len(list_obj)
		self.averaged_q = averaged_q
		self.alpha = obj.alpha
		if self.print_Running :
			print('Averaged_q=',averaged_q )




