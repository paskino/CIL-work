#%%
import numpy as np
import os
#load data
from cil.io import TXRMDataReader, TIFFWriter
from cil.processors import TransmissionAbsorptionConverter, Binner
from cil.plugins.tigre import FBP
from cil.utilities.display import show2D, show_geometry
#preprocessing
from cil.processors import Slicer
# PDHG reconstruction
from cil.optimisation.functions import MixedL21Norm, L2NormSquared, BlockFunction, IndicatorBox
from cil.optimisation.operators import BlockOperator, GradientOperator
from cil.plugins.tigre import ProjectionOperator
from cil.optimisation.algorithms import PDHG
# display
import matplotlib.pyplot as plt
# implicit 
from cil.plugins.ccpi_regularisation.functions import FGP_TV



def load_data(return2D=True):
    base_dir = os.path.abspath(r"C:\Users\ofn77899\Data\walnut")
    data_name = "valnut"
    filename = os.path.join(base_dir, data_name, "valnut_2014-03-21_643_28/tomo-A/valnut_tomo-A.txrm")
    # base_dir = os.path.abspath(r"C:\Users\ofn77899\Data\egg1" )
    # data_name = "gruppe 1"
    # filename = os.path.join(base_dir, data_name, "gruppe 1_2014-03-20_946_13/tomo-A/gruppe 1_tomo-A.txrm")

    data = TXRMDataReader(file_name=filename).read()
    if return2D:
        data = data.get_slice(vertical='centre') 
    else:
        binner = Binner(roi={'horizontal': (None, None, 4),'vertical': (None, None, 4)})
        data = binner(data)
    return data    

# show_geometry(data.geometry)
# use 1 over 5 angles

def preprocess(data, roi = None):
    if roi is not None:
        data = Slicer(roi=roi)(data)

    return TransmissionAbsorptionConverter()(data)

def run_adaptive_pdhg(algo, num_inner_iter, num_outer_iter):
    for i in range(num_outer_iter):
        algo.run(num_inner_iter)
        gamma = algo.solution.norm()
        sigma = gamma / (normK)
        tau = 1 / (gamma * normK)
        algo.sigma = sigma
        algo.tau = tau

def get_primal_loss(algo):
    return [a[0] for a in algo.loss]
#%%
if __name__== '__main__':
    data = load_data(return2D=True)
    data = preprocess(data, roi={'angle': (None, None, 20)})
    data.reorder(order='tigre')


    ig = data.geometry.get_ImageGeometry()

    fbp =  FBP(ig, data.geometry)
    recon = fbp(data)

    # define a TV reconstruction 
#%%
    # Explicit PDHG setup

    A = ProjectionOperator(ig, data.geometry)
    # Define BlockFunction F
    alpha_tv = 0.0003
    f1 = alpha_tv * MixedL21Norm()
    f2 = 0.5 * L2NormSquared(b=data)
    F = BlockFunction(f1, f2)

    # Define BlockOperator K
    Grad = GradientOperator(ig)
    K = BlockOperator(Grad, A)
    normK = K.norm()

    # Define Function G
    G = IndicatorBox(lower=0)



#%% 
    # create default PDHG
    algos = {}
    algo = PDHG(f = F, g = G, operator = K)
    algos['default'] = algo
    algo.max_iteration = 5000
    algo.update_objective_interval = 20
    algo.run(verbose=2)

    
#%% 
    # Find minimum of objective varying gamma around 1
    # sigma/tau = gamma**2
    # sigma * tau <= 1/normK**2

    norm_sol = algos['default'].solution.norm()

    gs = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, normK / norm_sol]
    obj_100 = []
    for g in gs:
        gamma = norm_sol * g
        sigma = gamma / (normK)
        tau = 1 / (gamma * normK)

        algo = PDHG(f = F, g = G, operator = K, 
                    max_iteration = 500,
                    update_objective_interval = 20, 
                    sigma = sigma, tau=tau)
        # pdhg_l1.y_old = K.range.allocate(1)
        algo.run(20, verbose=1)
        obj_100.append(algo.get_last_objective())

#%%

    plt.semilogx(gs[:-1], obj_100[:-1] , label='set gamma')
    plt.scatter(gs[-1], obj_100[-1], c='r', label='default')
    plt.xlabel('gamma')
    plt.ylabel('objective @ 20 iterations')
    plt.legend()
    plt.plot()
# %%

    # get the minimum in the list of the objective after 20 iterations
    dmin = np.argmin(obj_100)

    gamma = norm_sol * gs[dmin]
    sigma = gamma / (normK)
    tau = 1 / (gamma * normK)
    # Run "optimal gamma"
    algo = PDHG(f = F, g = G, operator = K, 
                max_iteration = 5000,
                update_objective_interval = 200, 
                sigma = sigma, tau=tau)
    algo.run(verbose=2)
    algos['explicit optimal gamma'] = algo
    


#%%
    # show2D([algo.solution, algos['default'].solution])
    # show2D(algo.solution - algos['optimal_gamma'].solution, cmap='seismic')
    # preconditioning

#%%
    ## Explicit adaptive gamma

    algo = PDHG(f = F, g = G, operator = K)

    algo.max_iteration = 5000
    algo.update_objective_interval = 20
    #%%
    # for i in range(40):
    #     algo.run(20)
    #     gamma = algo.solution.norm()
    #     sigma = gamma / (normK)
    #     tau = 1 / (gamma * normK)
    #     algo.sigma = sigma
    #     algo.tau = tau
    run_adaptive_pdhg(algo, 20, 50)
    algos['explicit adaptive'] = algo
#%%
    # explicit diagonal preconditioning

    S = K.range.allocate(1) / (K.direct(K.domain.allocate(1)))
    T = K.domain.allocate(1) / (K.adjoint(K.range.allocate(1)))

    # requires not to use axpby
    algo = PDHG(f = F, g = G, operator = K, sigma = S, tau = T,
        use_axpby=False)

    algo.max_iteration = 5000
    algo.update_objective_interval = 20


    # %%
    algo.run(5000, verbose=2)
    algos['explicit diagonal preconditioning'] = algo
# %%

    # implicit PDHG

    TV = (alpha_tv / ig.voxel_size_y) * FGP_TV(device='gpu', nonnegativity=True)

    algo = PDHG(f=f2, g=TV, operator=A)
    algo.max_iteration = 5000
    algo.update_objective_interval = 20
    algo.run(verbose=2)
    algos['implicit default'] = algo
# %%




    
    # show2D([algo.solution, adapt.solution], title=['default gamma', 'adaptive'])
    # # %%
    # show2D(algo.solution-adapt.solution, cmap='seismic', fix_range=(-0.002, 0.002))    
#%%
    # implicit adaptive

    algo = PDHG(f = f2, g = TV, operator = A)

    algo.max_iteration = 5000
    algo.update_objective_interval = 20


    # %%
    # for i in range(50):
    #     algo.run(20)
    #     gamma = algo.solution.norm()
    #     sigma = gamma / (normK)
    #     tau = 1 / (gamma * normK)
    #     algo.sigma = sigma
    #     algo.tau = tau
    run_adaptive_pdhg(algo, 20, 50)
    algos['implicit adaptive'] = algo
#%%

    # implicit diagonal preconditioning
    S = A.range.allocate(1) / (A.direct(A.domain.allocate(1)))
    T = A.domain.allocate(1) / (A.adjoint(A.range.allocate(1)))

    algo = PDHG(f = f2, g = TV, operator = A, sigma=S, tau=T)

    algo.max_iteration = 5000
    algo.update_objective_interval = 20
    algo.run(verbose=2)
    algos['implicit diagonal preconditioning'] = algo

 
# %%
    import matplotlib.pyplot as plt

    fig = plt.figure()
    for k,v in algos.items():
        plt.semilogy(v.iterations[1:], get_primal_loss(v)[1:], label=k )
    plt.xlabel('iterations')
    plt.ylabel('objective')
    plt.legend()
    plt.plot()
    # %%
