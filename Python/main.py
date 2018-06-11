import paths
from AMP_class import *
from StateEvolution_class import *


### AMP
def run_AMP(K,PW_choice,N,alpha,verbose):    
    AMP = ApproximateMessagePassing(K=K,PW_choice=PW_choice,N=N,alpha=alpha,seed=False,MC_activated=False,save_backup_running=False,print_Running=verbose)
    AMP.N_step_AMP = N_step_AMP
    AMP.initialization()
    AMP.AMP_iteration()
    return AMP
def plot_q(obj_AMP,obj_SE):
    obj_AMP.plot_q(obj_SE)
# Compute Generalization error AMP
def run_gen_error_AMP(obj,K,PW_choice,N,alpha,N_samples_gen_error): 
    gen_error , tab_gen_AMP = obj.gen_error(N_samples_gen_error)
    return tab_gen_AMP

def plot_gen_error(obj_AMP,tab_gen_AMP,gen_error_SE):
    obj_AMP.plot_gen_error(tab_gen_AMP,gen_error_SE)

### SE
def run_SE(K,PW_choice,alpha,verbose):
    SE = StateEvolution(K=K,PW_choice=PW_choice,alpha=alpha,channel='sign-sign',save_backup_running=False,seed=False,committee_symmetry=True,print_running=verbose) 
    SE.precision = 1e-5
    SE.initialization_mode = 5
    SE.initialization()
    SE.SE_iteration()
    return SE
def run_gen_error_SE(obj): 
    gen_error_SE = obj.gen_error()
    return gen_error_SE


### FOR THE DEMO 
N_step_AMP = 100 # max number of iterations
verbose = True


demo = True
if not demo : 
    K = 2 
    PW_choice = 'binary'
    N = 10000 
    alpha = 1.8

    obj_AMP = run_AMP(K,PW_choice,N,alpha,verbose)
    #obj_SE = run_SE(K,PW_choice,alpha,verbose)