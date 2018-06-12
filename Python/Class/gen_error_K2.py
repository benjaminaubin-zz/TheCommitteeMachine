import paths
from libraries import *

#import mcint
from skmonaco import mcquad, mcimport
from numpy import sign
class gen_err_MC(object):
	def __init__(self):
		self.borne_inf, self.borne_sup = -10 , 10
		self.N_iter_MCquad = 1000000
		self.NProcs = 1
		self.step = 0

	def initialize(self,qd,qa):
		self.qd = qd
		self.qa = qa

	# MC
	def sampler_mcquad(self,size):
		z =  (multivariate_normal.rvs([0,0,0,0], np.identity(4) , size))
		return z 
	def integrand_mcquad(self,X):
		qd , qa = self.qd , self.qa 
		x1 , x2 , x3 ,x4  = X[0] , X[1] , X[2] , X[3]
		term1 = sign( sign(x1)+sign(x2))
		term2 = sign( sign( (qa/2 + qd)*x1 + qa/2*x2 + x3 * sqrt(1-qa**2/2 -qa*qd -qd**2) ) + sign( qa/2*x1 +(qa/2+qd)*x2 -   qa *(qd + qa/2) / sqrt(1 - qa**2/2 -qa*qd -qd**2) * x3  + sqrt( ((1-qd**2) * (1- (qa+qd)**2 ) ) / (1-qa**2/2 - qa*qd - qd**2) ) * x4  ))
		return term1 * term2

	def gen_err_mcquad(self):
		result , error1 = mcimport(self.integrand_mcquad, self.N_iter_MCquad , self.sampler_mcquad ,nprocs = self.NProcs )
		return 1/2 - result
