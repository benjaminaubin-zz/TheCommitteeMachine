import paths

import warnings
warnings.filterwarnings("ignore")

import platform
if platform.system() == 'Darwin':
	import matplotlib as mpl
	mpl.use('TkAgg')

################################### LIBRARIES ###################################
######## MATH and SYSTEM ########
### System libraries
import sys , os , time , copy

### Mathematical libraries
import numpy as np 
import math , random
from math import pi,exp,sqrt,log,cosh,sinh,asin,acos,tanh

#import mpmath as mp
from scipy.stats import multivariate_normal , wishart
from scipy.integrate import quad, nquad, quadrature
from scipy.special import erfc , erf
from scipy.linalg import inv , sqrtm , det , norm 
from scipy.signal import savgol_filter as sgf
from scipy import real,imag 
from sklearn.datasets import make_spd_matrix
#from cmath import exp as cexp

## Multiprocessing
#from multiprocessing import Pool
#import itertools


### Load/Save data
import pickle
#import pandas as pd

### Monte Carlo
from skmonaco import mcquad, mcimport

import matplotlib.pyplot as plt
from matplotlib import rc
#import seaborn as sns

# Parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
Fontsize = 25
Linewidth = 1.5	
db,do,g,cr,mv,o,dc,dg= 'dodgerblue','darkorange','gold','crimson','mediumvioletred','orangered','darkcyan','darkgreen'



