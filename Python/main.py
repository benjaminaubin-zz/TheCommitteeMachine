import paths
from AMP_class import *
from StateEvolution_class import *


### AMP
def run_AMP(K,PW_choice,N,alpha,print_Running):    
    AMP = ApproximateMessagePassing(K=K,PW_choice=PW_choice,N=N,alpha=alpha,seed=False,MC_activated=False,save_backup_running=False,print_Running=print_Running)
    AMP.N_step_AMP = N_step_AMP
    AMP.initialization()
    AMP.AMP_iteration()
    return AMP
def plot_q(obj):
    obj.plot_q()
# Compute Generalization error AMP
def run_gen_error_AMP(obj,K,PW_choice,N,alpha,N_samples_gen_error): 
    gen_error = obj.gen_error(N_samples_gen_error)

### SE
def run_SE(K,PW_choice,alpha,print_Running):
    SE = StateEvolution(K=K,PW_choice=PW_choice,alpha=alpha,channel='sign-sign',save_backup_running=False,seed=False,committee_symmetry=True,print_running=print_Running) 
    SE.precision = 1e-5
    SE.initialization_mode = 5
    SE.initialization()
    SE.SE_iteration()
    return SE
def run_gen_error_SE(obj): 
    gen_error = obj.gen_error()
    

### FOR THE DEMO 
N_step_AMP = 250 # max number of iterations
print_Running = True


demo = True
if not demo : 
    K = 2 
    PW_choice = 'binary'
    N = 10000 
    alpha = 1.8

    obj_AMP = run_AMP(K,PW_choice,N,alpha,print_Running)
    #obj_SE = run_SE(K,PW_choice,alpha,print_Running)