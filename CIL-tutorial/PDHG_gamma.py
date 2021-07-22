#%%
import numpy as np
import os

from cil.io import TXRMDataReader, TIFFWriter
from cil.processors import TransmissionAbsorptionConverter, Binner
from cil.plugins.tigre import FBP
from cil.utilities.display import show2D, show_geometry

base_dir = os.path.abspath(r"C:\Users\ofn77899\Data\walnut")
data_name = "valnut"
filename = os.path.join(base_dir, data_name, "valnut_2014-03-21_643_28/tomo-A/valnut_tomo-A.txrm")
# base_dir = os.path.abspath(r"C:\Users\ofn77899\Data\egg1" )
# data_name = "gruppe 1"
# filename = os.path.join(base_dir, data_name, "gruppe 1_2014-03-20_946_13/tomo-A/gruppe 1_tomo-A.txrm")



is2D = True
data = TXRMDataReader(file_name=filename).read()
if is2D:
    data = data.get_slice(vertical='centre') 
else:
    binner = Binner(roi={'horizontal': (None, None, 4),'vertical': (None, None, 4)})
    data = binner(data)
    

# show_geometry(data.geometry)
# use 1 over 5 angles
from cil.processors import Slicer
data = Slicer(roi={'angle': (None, None, 5)})(data)

data = TransmissionAbsorptionConverter()(data)

data.reorder(order='tigre')

ig = data.geometry.get_ImageGeometry()

fbp =  FBP(ig, data.geometry)
recon = fbp(data)

# define a TV reconstruction 
#%%
from cil.optimisation.functions import MixedL21Norm, L2NormSquared, BlockFunction, IndicatorBox
from cil.optimisation.operators import BlockOperator, GradientOperator
from cil.plugins.tigre import ProjectionOperator
from cil.optimisation.algorithms import PDHG

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
algo = PDHG(f = F, g = G, operator = K)

algo.max_iteration = 1000
algo.update_objective_interval = 20
algo.run(verbose=2)

default_solution = algo.solution

#%% 
# Setup and run PDHG
norm_sol = algo.solution.norm()

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
import matplotlib.pyplot as plt

plt.semilogx(gs[:-1], obj_100[:-1] , label='predefined gamma')
plt.scatter(gs[-1], obj_100[-1], c='r', label='default')
plt.xlabel('gamma')
plt.ylabel('objective @ 20 iterations')
plt.legend()
plt.plot()
# %%

# get the minimum of the objective after 20 iterations
dmin = np.argmin(obj_100)

gamma = norm_sol * gs[dmin]
sigma = gamma / (normK)
tau = 1 / (gamma * normK)

algo = PDHG(f = F, g = G, operator = K, 
            max_iteration = 1000,
            update_objective_interval = 200, 
              sigma = sigma, tau=tau)
algo.run(verbose=2)

algodef = PDHG(f = F, g = G, operator = K, 
            max_iteration = 5000,
            update_objective_interval = 200)
algodef.run(verbose=2)



#%%
from cil.utilities.display import show2D
show2D([algo.solution, algodef.solution])
show2D(algo.solution - algodef.solution, cmap='seismic')
# preconditioning


# %%

# implicit PDHG
from cil.plugins.ccpi_regularisation.functions import FGP_TV

TV = (alpha_tv / ig.voxel_size_y) * FGP_TV(device='gpu', nonnegativity=True)

ialgo = PDHG(f=f2, g=TV, operator=A)
ialgo.max_iteration = 5000
ialgo.update_objective_interval = 20
ialgo.run(20, verbose=2)
# %%

## adaptive gamma
adapt = PDHG(f = F, g = G, operator = K)

adapt.max_iteration = 5000
adapt.update_objective_interval = 20
#%%
for i in range(5):
    adapt.run(20)
    gamma = adapt.solution.norm()
    sigma = gamma / (normK)
    tau = 1 / (gamma * normK)
    adapt.sigma = sigma
    adapt.tau = tau




# %%
import matplotlib.pyplot as plt

fig = plt.figure()
plt.semilogy(algo.iterations[1:], [a[0] for a in algo.loss[1:]] , label='default gamma')
plt.semilogy(adapt.iterations[1:], [a[0] for a in adapt.loss[1:]], c='r', label='adaptive')
plt.xlabel('iterations')
plt.ylabel('objective')
plt.legend()
plt.plot()
# %%

show2D([algo.solution, adapt.solution], title=['default gamma', 'adaptive'])
# %%
show2D(algo.solution-adapt.solution, cmap='seismic', fix_range=(-0.002, 0.002))    
# %%
