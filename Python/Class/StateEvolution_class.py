from gout_sign_sign import *
from gen_error_K2 import *
 
################################## SAVE OBJECT ##################################
def save_object(obj,filename_object):
	with open(filename_object, 'wb') as output:
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
	print('Object saved:',filename_object)
def load_object(filename_object,print_load=True):
	with open(filename_object, 'rb') as input:
		obj = pickle.load(input)
	if print_load :
		print("Object loaded:",filename_object)
	return obj

############################# STATE EVOLUTION CLASS #############################
class StateEvolution(object):

	def __init__(self,K=2,PW_choice='binary',alpha=1,channel='sign-sign',save_backup_running=False,seed=False,committee_symmetry=False,print_running=True,initialization_mode='load_backup'):

		## Parameters
		self.K = K # number of hidden units
		self.alpha = alpha 
		self.channel = channel 

		## Committee symmetry
		self.committee_symmetry = committee_symmetry
		
		## For sign - sign
		if self.channel == 'sign-sign':
			if self.K % 2 == 0 : # K pair
				self.y_values = [-1 , 0 , 1]
			elif self.K %2 == 1: # K impair
				self.y_values = [-1 , 1]
		elif self.channel == 'linear-relu':
			self.y_values = [0 , 1] 

		## Initialization_mode
		if initialization_mode == 'load_backup':
			self.initialization_mode = 3
		elif initialization_mode == 'committee_symmetry': 
			self.initialization_mode = 2
		else : 
			self.initialization_mode = 1
		
		## multiprocessing
		self.use_pool = False

		## Weights
		self.W1 = np.ones((K)) # vector with only ones !
		self.PW_choice = PW_choice # 'binary' or 'gaussian'
		self.sigma_W = 1 

		## Integration
		self.borne_sup = 10
		self.borne_inf = -10
		self.epsabs = 1e-7
		
		## Storage of the matrices
		# For committee symmetry
		self.qd = 0.1
		self.qa = 0.5
		self.qhatd = 0.1
		self.qhata = 0.5

		self.Q = np.identity(self.K) # Fixed for binary !! 
		self.q = 0.1 * np.identity(self.K)
		self.qhat = 0.1 * np.identity(self.K) 
		# q
		self.q_inv = inv(self.q)
		self.q_sqrt = sqrtm(self.q)
		self.q_sqrt_inv = inv(self.q_sqrt)
		# Q-q
		self.Q_q = self.Q - self.q
		self.Q_q_inv = inv(self.Q_q)
		self.Q_q_sqrt = sqrtm(self.Q_q)
		self.Q_q_sqrt_inv = inv(self.Q_q_sqrt)
		# qhat
		self.qhat_inv = inv(self.qhat)
		self.qhat_sqrt = sqrtm(self.qhat)
		self.qhat_sqrt_inv = inv(self.qhat_sqrt)

		## Convergency SE equations
		self.step = 0
		self.difference = 1
		self.precision = 5*1e-5
		self.N_step_SE = 100
		
		## damping
		self.damping_coef = 1
		self.damping_coef_slow = 1
		
		## condition to break
		self.threshold_q_stop = 0.999

		## Save data 
		self.save_backup_running = save_backup_running		
		self.path_BackupObj = 'Data/BackupObj_SE_'+ self.PW_choice +'_K='+ str(K)+'/' 
		#if not os.path.exists(self.path_BackupObj):
		#	os.makedirs(self.path_BackupObj) 
		self.filename_object = self.path_BackupObj + 'SE_backupObject_'+PW_choice+'_K='+str(K)+'_alpha='+'%.4f'% alpha+'.pkl'
	
		self.path_Figures = 'Figures/'
		self.saveFig = False
		self.plotFig = True
		self.print_running = print_running

		self.evolution_q = []
		self.evolution_qhat = []

		# Seed 
		self.use_seed = seed
		if self.use_seed :
			np.random.seed(0)

		self.time_start = time.time()
		self.step_integration = 0

	################################################ STATE EVOLUTION ################################################
	
	############## Initialisation ##################
	def initialization(self):
		print('Start initialization')
		test_spd = False
		while not test_spd:
			# using sklearn
			if self.initialization_mode == 1:
				self.q = make_spd_matrix(self.K)
				self.q *= coef_q/np.amax(self.q)
				self.qhat = make_spd_matrix(self.K)
				self.qhat *= coef_qhat/np.amax(self.qhat)
			if self.initialization_mode == 2 : 
				self.q = self.qd * np.identity(self.K) + self.qa / self.K * np.ones((self.K,self.K))
				self.qhat = self.qhatd * np.identity(self.K) + self.qhata / self.K * np.ones((self.K,self.K))
			if self.initialization_mode == 3 :
				try :
					path = 'Python/Data/BackupObj_SE_'+ self.PW_choice +'_K='+ str(self.K)+'/'
					self.get_all_backup(path,print_load=False)
					tab = self.tab_alpha
					index = (np.abs(tab-self.alpha)).argmin()
					print('Succeeded to load old objects : ','Nearest alpha=',tab[index])
					self.q = self.tab_q[:,:,index]
					self.qhat = self.tab_qhat[:,:,index]
				except :
					print('Failed to load old objects') 
					qd = 0.25
					qa = 0.05
					self.q = qd * np.identity(self.K) + qa * np.ones((self.K,self.K))
					self.qhat = qd * np.identity(self.K) + qa * np.ones((self.K,self.K))
 
			self.update_all_q_qhat_functions()
			V_inv = self.Q_q_inv
			test_spd = det(V_inv) > 0
			if not test_spd :
				print('Wrong initialization')
			else : 
				print('Successful initialization','\n')
	def update_all_q_qhat_functions(self):
		self.q_inv = inv(self.q)
		self.q_sqrt = sqrtm(self.q)
		self.q_sqrt_inv = inv(self.q_sqrt)

		self.Q_q = self.Q - self.q
		self.Q_q_inv = inv(self.Q_q)
		self.Q_q_sqrt = sqrtm(self.Q_q)
		self.Q_q_sqrt_inv = inv(self.Q_q_sqrt)

		self.qhat_inv = inv(self.qhat)
		self.qhat_sqrt = sqrtm(self.qhat)
		self.qhat_sqrt_inv = inv(self.qhat_sqrt)
	def damping(self,A_new,A_old):
		res =  A_old * (1 - self.damping_coef) + A_new * self.damping_coef
		return res
	############## State Evolution (SE) ############
	def SE_step(self):
		start = time.time() 
		q , qhat = np.zeros((self.K,self.K)) , np.zeros((self.K,self.K))
		
		if self.use_pool :
			pool = Pool(self.K * self.K)
			q = pool.starmap(self.derivative_IW, list(itertools.product(range(self.K),range(self.K))))
			pool.close()
			pool = Pool(self.K * self.K)
			qhat = pool.starmap(self.derivative_IZ, list(itertools.product(range(self.K),range(self.K))))
			pool.close()

			q = 2 * np.array(q).reshape(self.K,self.K)
			qhat = 2 * self.alpha * np.array(qhat).reshape(self.K,self.K)

		else :
			
			if self.committee_symmetry :
				qd = 2 * self.derivative_IW(0,0)
				qhatd = 2 * self.alpha * self.derivative_IZ(0,0)
				qa = 2 * self.derivative_IW(0,1)
				qhata = 2 * self.alpha * self.derivative_IZ(0,1)
				q = (qd-qa) * np.identity(self.K) + qa * np.ones((self.K,self.K)) 
				qhat = (qhatd-qhata) * np.identity(self.K) + qhata * np.ones((self.K,self.K))  
			
			else :
				for r in range(0,self.K):
					for r_p in range(0,r+1) :
						q[r,r_p] = 2 * self.derivative_IW(r,r_p)
						qhat[r,r_p] = 2 * self.alpha * self.derivative_IZ(r,r_p)

						# Symmetric matrices X[r_p,r] = X[r,r_p]
						q[r_p,r] = q[r,r_p]
						qhat[r_p,r] = qhat[r,r_p]

		end = time.time() 
		# Damping 
		self.q = self.damping(q , copy.copy(self.q) )
		self.qhat = self.damping(qhat , copy.copy(self.qhat) )
		
		# Update matrices
		self.update_all_q_qhat_functions()
		#self.print_all()

		return (q , qhat)
	def SE_iteration(self):
		self.difference , self.step , stop = 10 , 0 , False
		# Print parameters
		print('K=',self.K,'PW=',self.PW_choice,'alpha=',self.alpha,'channel=',self.channel)
		
		self.start = time.ctime()
		print('Start SE:',self.start)
		self.save_evolution()

		while self.step < self.N_step_SE and stop == False and self.difference > self.precision:
			if not self.check_def_pos() :
				# break if not definite positive
				break
			q_tmp = copy.copy(self.q)
			self.step += 1 
			## One step SE
			self.SE_step()
			self.save_evolution()

			## Break if q > 0.99
			stop = np.any(self.q > self.threshold_q_stop)
			if stop : 
				print('BREAK SE: q > threshold - perfect generalization')

			if np.any(self.q > 0.9) and self.damping_coef!=self.damping_coef_slow :
				print('New damping !')
				self.damping_coef = self.damping_coef_slow 

			## Difference with previous step
			self.difference = self.matrix_difference(q_tmp)
			if self.print_running:
				print('Step = ',self.step,'diff q =',self.difference)

			if self.save_backup_running :
				save_object(self,self.filename_object)

		self.print_matrix(self.q,'Final overlap q_SE=')
		print('End SE')

	############### ANNEX FUNCTIONS ################
	def print_matrix(self,M,txt):
		print(txt)
		n,m = M.shape
		for i in range(n):
			print(M[i,:])
	def matrix_difference(self,q_tmp):
		res = np.sum(np.abs(self.q - q_tmp))
		return res 
	def check_def_pos(self):
		test = True
		if det(self.q)<=0:
			print('q not definite positive! det q=',det(self.q))
			test = False
		if det(self.qhat )<=0:
			print('qhat not definite positive! det qhat=',det(self.qhat))
			test = False
		return test
	def save_evolution(self):
		self.evolution_q.append(self.q)
		self.evolution_qhat.append(self.qhat)
	def print_all(self):
		self.print_matrix(self.q,'q=')
		self.print_matrix(self.qhat,'qhat=')
		self.print_matrix(self.q_inv,'q_inv=')
		self.print_matrix(self.q_sqrt,'q_sqrt=')
		self.print_matrix(self.q_sqrt_inv,'q_sqrt_inv=')
		self.print_matrix(self.Q_q,'Q_q=')
		self.print_matrix(self.Q_q_inv,'Q_q_inv=')
		self.print_matrix(self.Q_q_sqrt,'Q_q_sqrt=')
		self.print_matrix(self.Q_q_sqrt_inv,'Q_q_sqrt_inv=')
		self.print_matrix(self.qhat_inv,'qhat_inv=')
		self.print_matrix(self.qhat_sqrt,'qhat_sqrt=')
		self.print_matrix(self.qhat_sqrt_inv,'qhat_sqrt=')

	################# SP Equations #################
	
	############ Derivative Integral IW ############
	def derivative_IW(self,r,r_p):
		# Binary weights
		if self.PW_choice == 'binary':
			if self.K == 1 :
				resul = self.derivative_IW_integral_binary_K1(r,r_p)
			elif self.K == 2 :
				resul = self.derivative_IW_integral_binary_K2(r,r_p)
			else : 
				print('No prior for binary and K>2')
		# Gaussian weights
		elif self.PW_choice == 'gaussian' :
			resul = self.derivative_IW_gaussian()[r,r_p]
		# Otherwise
		else :
			print('No prior defined')
			resul = 0
		return resul
	##### Binary
	# Configurations and integrals JW, JW0
	def configuration_W(self,n):
		W = np.array( [int(x) for x in list('{0:0b}'.format(n).zfill(self.K)) ] )
		W = 2*W-1
		return W
	def integrals_f0_g0(self,xi,r,r_p):
		if self.K == 1 : 
			f0 = exp(-1/2*self.qhat) * 2*cosh(xi*sqrt(self.qhat)) / 2
			df0 = exp(-1/2*self.qhat) * 2*sinh(xi*sqrt(self.qhat)) / 2
			g0 = 1/f0 * df0
			return (f0 , g0) 
		if self.K >= 2 : 
			f0 , df0 = 0 , 0
			for i in range(2**self.K) :
				W = self.configuration_W(i)
				argument = - 1/2 * W.transpose().dot(self.qhat).dot(W) + xi.transpose().dot(self.qhat_sqrt).dot(W) # checked
				weight = exp(argument)
				f0 += weight
				df0 += weight * W
			f0 /= (2**self.K)
			df0 /= (2**self.K)
			g0 = 1/f0 * df0 
			return (f0 , g0)
	## dIW for K=1
	# New with g0, f0
	def derivative_IW_integral_binary_K1(self,r,r_p):
		resul_integral = quad(self.integrand_derivative_IW_integral_binary_K1, self.borne_inf , self.borne_sup  , args=(r,r_p))[0]
		resul = 1/2 * resul_integral
		return resul
	def integrand_derivative_IW_integral_binary_K1(self,xi1,r,r_p):
		xi = np.array([xi1])
		gaussian_measure = exp(-1/2 * xi.transpose().dot(xi) ) / (2*pi)**(self.K/2)
		( f0, g0 ) = self.integrals_f0_g0(xi,r,r_p)
		resul = gaussian_measure * f0 * g0**2
		return resul
	## dIW for K=2
	def derivative_IW_integral_binary_K2(self,r,r_p):
		resul_integral = nquad(self.integrand_derivative_IW_integral_binary_K2, [ [self.borne_inf , self.borne_sup ] , [self.borne_inf , self.borne_sup ] ] , args=(r,r_p),opts={'epsabs':self.epsabs})[0]
		resul = 1/2 *resul_integral
		return resul
	def integrand_derivative_IW_integral_binary_K2(self,xi1,xi2,r,r_p):
		xi = np.array([xi1,xi2])
		gaussian_measure = exp(-1/2 * xi.transpose().dot(xi) ) / (2*pi)**(self.K/2)
		( f0, g0 ) = self.integrals_f0_g0(xi,r,r_p)
		g0 = g0.reshape(1,self.K)
		g0_squared = g0.transpose().dot(g0)
		resul =  gaussian_measure * f0 * ( g0_squared[r,r_p])
		return resul
	##### Gaussian
	# dIW for all K
	def derivative_IW_gaussian(self):
		tmp = inv( np.identity(self.K) + self.sigma_W * self.qhat)
		resul = self.sigma_W/2  * ( np.identity(self.K)  - tmp.transpose())
		return resul 
	def gaussian_measure(self,t):
		resul = exp(-t**2/2)/sqrt(2*pi)
		return resul

	############ Derivative Integral IZ ############
	def derivative_IZ(self,r,r_p):
		mode_all_y = False # Symmetric with respect to y -> -y
		if mode_all_y : 
			y_values = self.y_values
			dIZ = 1/2 * np.sum([self.derivative_IZ_y(y,r,r_p) for y in y_values])
		else : 
			# speed up because y=1 idem than y=-1
			y_values = [ y for y in self.y_values if y >=0 ]
			dIZ = 0 
			for y in y_values : 
				if y >0 :
					dIZ += 1/2 * 2 * self.derivative_IZ_y(y,r,r_p) # term for y = 1 idem that for y= -1
				else :
					dIZ += 1/2 * self.derivative_IZ_y(y,r,r_p)
		return dIZ
	def derivative_IZ_y(self,y,r,r_p):
		# K == 1
		if self.K ==1 :
			res = quad(self.integrand_derivative_IZ_y_argument_K1, self.borne_inf , self.borne_sup  , args=(y,r,r_p,))[0]
		# K == 2
		elif self.K ==2 : 
			# K == 2 and sign-sign
			if self.channel == 'sign-sign':
				res = nquad(self.integrand_derivative_IZ_y_argument_K2, [ [self.borne_inf , self.borne_sup ] , [self.borne_inf , self.borne_sup ] ] , args=(y,r,r_p,))[0]
				
			# K == 2 and linear-relu
			elif self.channel == 'linear-relu':
				derivative_IZ_relu = class_derivative_IZ_relu(self.K ,q_sqrt = self.q_sqrt , V_inv = self.Q_q_inv)
				res = derivative_IZ_relu. derivative_IZ_y(y,r,r_p)
		else:
			print('Channel not defined')
		return res
	def integrand_derivative_IZ_y_argument_K2(self,xi1,xi2,y,r,r_p):
		xi = np.array([xi1,xi2])
		gaussian_measure = exp(-1/2* xi.transpose().dot(xi) ) / (2*pi)**(self.K/2) 
		fout , gout = self.integrals_fout_gout(y,xi,r,r_p)
		gout = gout.reshape(1,self.K)
		gout_squared = gout.transpose().dot(gout)
		res =  gaussian_measure * fout * ( gout_squared [r,r_p])
		return res
	def integrand_derivative_IZ_y_argument_K1(self,xi,y,r,r_p):
		gaussian_measure = exp(-1/2* xi**2 ) / (2*pi)**(self.K/2) 
		fout , gout = self.integrals_fout_gout(y,xi,r,r_p)
		res =  gaussian_measure * fout * ( gout**2)
		return res
	def integrals_fout_gout(self,y,xi,r,r_p):
		V = self.Q_q
		V_inv = self.Q_q_inv
		omega = self.q_sqrt.dot(xi)	

		if self.channel == 'sign-sign':
			if self.K <=2 : 
				gout_ss = gout_sign_sign(K=self.K,y=y,omega=omega,V=V,V_inv=V_inv)
				(fout, gout) = gout_ss.fout_gout()
			else : 
				gout_lr = gout_sign_sign_complex(K=self.K,y=y,omega=omega,V=V,V_inv=V_inv)
				(fout, gout) = gout_lr.fout_gout()

		else:
			print('No gout defined')

		return (fout,gout)
	def integrals_fout(self,y,xi,r,r_p):
		V = self.Q_q
		V_inv = self.Q_q_inv
		omega = self.q_sqrt.dot(xi)	

		if self.channel == 'sign-sign':
			if self.K <=2 : 
				gout_ss = gout_sign_sign(K=self.K,y=y,omega=omega,V=V,V_inv=V_inv)
				fout= gout_ss.fout()
			else : 
				gout_lr = gout_sign_sign_complex(K=self.K,y=y,omega=omega,V=V,V_inv=V_inv)
				fout = gout_lr.fout()
		else:
			print('no fout defined')
		return (fout)

	################# Free energy #################
	## PHI
	def Phi(self,q,qhat):
		self.q = q
		self.qhat = qhat
		self.update_all_q_qhat_functions()
		res = -1/2 * np.trace((q).dot(qhat)) +self.IW()  + self.alpha *  self.IZ() 
		return res
	## IW 
	def IW(self):
		if self.K == 1 :  
			res = quad(self.integrand_IW_K1,  self.borne_inf , self.borne_sup  , args=())[0]
		elif self.K == 2 :  
			res = nquad(self.integrand_IW_K2, [ [self.borne_inf , self.borne_sup ] , [self.borne_inf , self.borne_sup ] ] , args=())[0]
		else : 
			print('IW not defined for K=',self.K)
		return res
	def integrand_IW_K1(self,xi):
		xi = np.array([xi])
		gaussian_measure = exp(-1/2 * xi.transpose().dot(xi) ) / (2*pi)**(self.K/2)
		( f0, _ ) = self.integrals_f0_g0(xi,0,0)
		if f0 == 0 : 
			res = 0 
		else : 
			res = gaussian_measure * f0 * log(f0)
		return res
	def integrand_IW_K2(self,xi1,xi2):
		xi = np.array([xi1,xi2])
		gaussian_measure = exp(-1/2 * xi.transpose().dot(xi) ) / (2*pi)**(self.K/2)
		( f0, _ ) = self.integrals_f0_g0(xi,0,0)
		if f0 == 0 : 
			res = 0 
		else : 
			res = gaussian_measure * f0 * log(f0)
		return res
	## IZ
	def IZ(self):
		y_values = self.y_values
		res = np.sum([self.IZ_y(y) for y in y_values])
		return res
	def IZ_y(self,y):
		if self.K == 1 : 
			res = quad(self.integrand_IZ_K1, self.borne_inf , self.borne_sup  , args=(y))[0]
		elif self.K ==2 :
			res = nquad(self.integrand_IZ_K2, [ [self.borne_inf , self.borne_sup ] , [self.borne_inf , self.borne_sup ] ] , args=(y,))[0]
		else : 
			print('IZ not defined for K=',self.K)
		return res
	def integrand_IZ_K1(self,xi,y): 
		gaussian_measure = exp(-1/2* xi**2 ) / (2*pi)**(self.K/2) 
		fout = self.integrals_fout(y,xi,0,0)
		if fout == 0 :
			res = 0
		else : 
			res =  gaussian_measure * fout * log(fout)
		return res
	def integrand_IZ_K2(self,xi1,xi2,y): 
		xi = np.array([xi1,xi2])
		gaussian_measure = exp(-1/2* xi.transpose().dot(xi) ) / (2*pi)**(self.K/2) 
		fout = self.integrals_fout(y,xi,0,0)
		if fout == 0 :
			res = 0
		else : 
			res =  gaussian_measure * fout * log(fout)
		return res
	# PHI perfect generalization
	def Phi_q1(self):	
		res = -self.K * log(2)
		return res

	################# Generalization error #################
	def gen_error(self):
		qa = self.K * self.q[0,1]
		qd = self.q[0,0] - qa/self.K
		gen_err_MC_obj = gen_err_MC()
		gen_err_MC_obj.initialize(qd,qa)
		gen_err = 1/2 * gen_err_MC_obj.gen_err_mcquad()
		print('Generalization Error=',gen_err)
		return gen_err

	################################################## PLOTS ##################################################
	def get_all_backup(self,path_data,print_load=True):
		os.chdir(path_data)
		list_files = sorted(sorted([s for s in os.listdir() if s.endswith('.pkl')], key=os.path.getmtime))
		step = 0
		tab_q = np.zeros((self.K,self.K,len(list_files)))
		tab_qhat = np.zeros((self.K,self.K,len(list_files)))
		tab_alpha = np.zeros(len(list_files))

		for file in list_files:
			SE_backup = load_object(file,print_load)
			alpha = SE_backup.alpha
			tab_q[:,:,step] = SE_backup.q
			tab_qhat[:,:,step] = SE_backup.qhat
			tab_alpha[step] = alpha
			step +=1
		self.tab_q = tab_q
		self.tab_qhat = tab_qhat
		self.tab_alpha = tab_alpha
		os.chdir('../../../')
		#print(os.getcwd())
	def plot_MSE_q_SE(self):
		self.saveFig = True
		title = r'$\textrm{SE: } \textbf{q}(\alpha) \textrm{ and MSE }(\alpha)  \textrm{ alpha PW=}$'+self.PW_choice+' K='+str(self.K)
		title_save = 'SE_alpha_PW='+self.PW_choice+'_K'+str(self.K)+'.eps'

		db,do,g,cr,mv,o,dc,dg= 'dodgerblue','darkorange','gold','crimson','mediumvioletred','orangered','darkcyan','darkgreen'
		colors = np.array([ [dc, cr ] ,[db , do ]  ])
		Linewidth = 1
		Fontsize = 20

		#sns.set()
		fig, ax1 = plt.subplots(figsize=[8,8])
		ax2 = ax1.twinx()
		ax2.grid(False)
		tab_l = []

		# Data
		tab_q = self.tab_q
		tab_alpha = self.tab_alpha

		# MSE 
		N = tab_q.shape[2]
		Q_tiled = np.zeros((self.K,self.K,N))
		for i in range(N):
			Q_tiled[:,:,i] = self.Q 
		MSE_mat = 1/self.K*( Q_tiled - tab_q)
		MSE = np.zeros(N)
		for i in range(N):
			MSE[i]  = np.trace(MSE_mat[:,:,i])
		# Plot MSE
		l, = ax2.plot(tab_alpha,MSE,'.',label='MSE',Linewidth=1.5,color=mv) 
		tab_l.append(l)

		if self.PW_choice == 'binary':
			#  Optimal error
			file = 'Data/' + 'SE_FreeEnergy_'+self.PW_choice+'_K='+str(self.K)+'.pkl'
			Phi = load_object(file)
			tab_Phi = -np.array(Phi.tab_Phi)
			tab_Phi_q1 = -np.array(Phi.tab_Phi_q1)
			print(tab_Phi)
			tab = [tab_Phi-tab_Phi_q1<0][0]
			ind_transition = np.where(tab==False)[0][0]
			print('alpha=',Phi.tab_alpha[ind_transition])
			optimal_MSE = np.multiply(MSE,tab)

			l, = ax2.plot(tab_alpha,optimal_MSE,'--',label='Optimal',Linewidth=1.5,color='k') 
			tab_l.append(l)

		# Overlap
		for i in range(self.K):
			for j in range(self.K):
				data = tab_q[i,j,:]
				l, = ax1.plot(tab_alpha,data,'.',color=colors[i,j],Markersize =5, Linewidth=Linewidth,label=r'$q$['+str(i)+','+str(j)+']')
				tab_l.append(l)

		labs = [l.get_label() for l in tab_l]
		plt.legend(tab_l, labs, loc=[0,0.4], fontsize=Fontsize)

		# Title
		plt.title(title,fontsize=Fontsize)
		# Label axis
		ax1.set_xlabel(r'$\alpha$',fontsize=Fontsize)
		ax2.set_ylabel(r'$MSE$',fontsize=Fontsize)
		ax1.set_ylabel(r'$\textbf{q}$',fontsize=Fontsize)
		# Limits axis
		ax1.set_xlim([0,max(tab_alpha)])
		ax1.set_ylim([-0.001,1.001])
		ax2.set_ylim([-0.001,1.001])

		if self.saveFig:
			if not os.path.exists(self.path_Figures):
				os.makedirs(self.path_Figures)
			plt.savefig(self.path_Figures+title_save, format='eps', dpi=800,bbox_inches="tight")

		if self.plotFig:
			plt.show(block=False)
			input("Press Enter to continue...")
			plt.close()
	
		if self.PW_choice == 'binary':
			fig0, ax0 = plt.subplots(figsize=[8,8])
			ax0.grid(False)
			ax0.plot(tab_alpha,tab_Phi,color = mv, label='AMP/SE Fixed point')
			ax0.plot(tab_alpha,tab_Phi_q1,color ='k',label='Optimal Fixed point')
			ax0.set_xlabel(r'$\alpha$',fontsize=Fontsize)
			ax0.set_ylabel(r'$-\Phi$',fontsize=Fontsize)
			min_x , max_x = 0 , max(tab_alpha)
			min_y , max_y = np.amin(np.concatenate([tab_Phi,tab_Phi_q1])) , np.amax(np.concatenate([tab_Phi,tab_Phi_q1]))
			ax0.set_xlim([min_x , max_x])
			ax0.set_ylim([min_y,max_y])
			plt.legend(fontsize=Fontsize)
			plt.show(block=False)
			input("Press Enter to continue...")
			plt.close()
	def plot_q(self):
		title = r'SE - $q_{SE}(t)$ at $\alpha=$'+str(self.alpha)+' K='+str(self.K)
		Fontsize = 25
		Fontsize_ticks = 20

		fig, ax1 = plt.subplots(figsize=[8,8])
		colors = np.array([ [dc, cr ] ,[db , do ]   ])

		n = len(self.evolution_q);
		tab_q = np.array(self.evolution_q);
		tab_t = np.arange(0,n,1);

		for i in range(self.K):
			for j in range(self.K):
				data = tab_q[:,i,j];
				ax1.plot(tab_t,data,'-o',color=colors[i,j],Markersize =3, Linewidth=1.5,label=r'$q$['+str(i)+','+str(j)+']')

		ax1.set_xlabel(r'time $t$',fontsize=Fontsize)
		ax1.set_ylabel(r'$q$',fontsize=Fontsize)
		plt.legend(loc='best', fontsize=Fontsize)
		plt.title(title,fontsize=Fontsize)
		ax1.set_ylim([np.amin(tab_q),np.amax(tab_q)])
		ax1.set_xlim([0,max(tab_t)])
		ax1.tick_params(labelsize=Fontsize_ticks)
		plt.show()





