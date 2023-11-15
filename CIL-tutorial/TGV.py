#%%
from cil.optimisation.functions import BlockFunction
from cil.optimisation.functions import L1Norm
from cil.optimisation.functions import MixedL21Norm
from cil.optimisation.functions import ZeroFunction
from cil.optimisation.operators import BlockOperator
from cil.optimisation.operators import ZeroOperator
from cil.optimisation.operators import GradientOperator
from cil.optimisation.operators import IdentityOperator
from cil.optimisation.operators import SymmetrisedGradientOperator
from cil.optimisation.algorithms import PDHG
from cil.io import NEXUSDataWriter
from cil.io import NEXUSDataReader
from cil.utilities import dataexample
from cil.utilities.display import show2D
from cil.plugins.ccpi_regularisation.functions import TGV

#%%

N = 64
M = N
data = dataexample.CAMERA.get(size=(N, M))

show2D(data)
#%%
tgv = TGV()
a = tgv.proximal(data, 0.1)

show2D([data, a])

#%%
# Setup PDHG for TGV regularisation
alpha = 0.10
# gamma = alpha/beta
# beta =  alpha/gamma
gamma = 1.0

beta = alpha / gamma

noisy_data = data
# Define BlockFunction f
f1 = L1Norm(b=noisy_data)
f2 = alpha * MixedL21Norm()
f3 = beta * MixedL21Norm() 
f = BlockFunction(f1, f2, f3)         

# Define function g 
g = ZeroFunction()

# Define BlockOperator K
ig = data.geometry
MO = IdentityOperator(ig)

#%%                                                                         
K11 = MO
K21 = GradientOperator(ig)
K32 = SymmetrisedGradientOperator(K21.range)
K12 = ZeroOperator(K32.domain, ig)
K22 = IdentityOperator(K21.range)
K31 = ZeroOperator(ig, K32.range)
K = BlockOperator(K11, K12, K21, -K22, K31, K32, shape=(3,2) )

# Compute operator Norm
normK = K.norm()
sigma = 1.
tau = 1./(sigma*normK**2)

# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=K,
            max_iteration = 100, 
            # sigma=sigma, tau=tau,
            update_objective_interval = 100)
pdhg.run(verbose = 2)    

#%%
# 
show2D([a, pdhg.solution[0]])
# %%
