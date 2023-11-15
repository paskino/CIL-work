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
from cil.optimisation.functions import L2NormSquared
import numpy as np
import scipy.sparse as sp
import cvxpy
#%%

N = 64
M = N
data = dataexample.CAMERA.get(size=(N, M))

show2D(data)
#%%
# Setup PDHG for TGV regularisation
alpha = 0.10
# gamma = alpha/beta
# beta =  alpha/gamma
gamma = 1.0

beta = alpha / gamma

#%%
tgv = TGV(alpha=alpha, gamma=gamma, max_iteration=1000)
a = tgv.proximal(data, 1.0)

show2D([data, a])

#%%

noisy_data = data
# Define BlockFunction f

f1 = 0.5 * L2NormSquared(b=noisy_data)
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
sigma = 1./normK
tau = 1./normK
#%%
# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=K,
            max_iteration = 1000, 
            # sigma=sigma, tau=tau,
            update_objective_interval = 100)
pdhg.run(1000, verbose = 2, print_interval = 100)    

tv_cil = pdhg.solution[0]
#%%
# 
show2D([a, pdhg.solution[0], a - pdhg.solution[0]])
# %%

def tv_cvxpy_regulariser(u, isotropic=True, direction = "forward", boundaries = "Neumann"):

    G = sparse_gradient_matrix(u.shape, direction = direction, order = 1, boundaries = boundaries)   

    DX, DY = G[1], G[0]

    if isotropic:
        return cvxpy.sum(cvxpy.norm(cvxpy.vstack([DX @ cvxpy.vec(u), DY @ cvxpy.vec(u)]), 2, axis = 0))
    else:
        return cvxpy.sum(cvxpy.norm(cvxpy.vstack([DX @ cvxpy.vec(u), DY @ cvxpy.vec(u)]), 1, axis = 0)) 

# Create gradient operator as a sparse matrix that will be used in CVXpy to define Gradient based regularisers
def sparse_gradient_matrix(shape, direction='forward', order=1, boundaries='Neumann', **kwargs):
    
    len_shape = len(shape)    
    allMat = dict.fromkeys(range(len_shape))
    discretization = kwargs.get('discretization',[1.0]*len_shape)

    if order == 1:

        # loop over the different directions
        for i in range(0,len_shape):

            if direction == 'forward':
            
                # create a sparse matrix with -1 in the main diagonal and 1 in the 1st diagonal
                mat = 1/discretization[i] * sp.spdiags(np.vstack([-np.ones((1,shape[i])),np.ones((1,shape[i]))]), [0,1], shape[i], shape[i], format = 'lil')

                # boundary conditions
                if boundaries == 'Neumann':
                    mat[-1,:] = 0
                elif boundaries == 'Periodic':
                    mat[-1,0] = 1

            elif direction == 'backward':

                # create a sparse matrix with -1 in the -1 and 1 in the main diagonal
                mat = 1/discretization[i] * sp.spdiags(np.vstack([-np.ones((1,shape[i])),np.ones((1,shape[i]))]), [-1,0], shape[i], shape[i], format = 'lil')

                # boundary conditions
                if boundaries == 'Neumann':
                    mat[:,-1] = 0
                elif boundaries == 'Periodic':
                    mat[0,-1] = -1

            # Use Kronecker product to compute the full sparse matrix representing the GradientOperator. This will be applied
            # to a "flatten" array, i.e., a vector and "the reshaped" result describes the forward/backward differences for all
            # the directions from "len_shape" and under different boundary conditions, e.g., Neumann and Periodic.

            # The difference with the GradientOperator.py and FiniteDifferenceOperator.py is that we do not store in memory a matrix
            # in order to compute the matrix-vector multiplication "A*x". This is also known as "matrix free" optimisation problem.
            # However, to set up and use the API of CVXpy, we need this matrix representation of a linear operator such as the GradientOperator.
            # 
            # The following constructs the finite complete difference matrix for all (len_shape) dimensions and store the ith finite 
            # difference matrix in allmat[i].  In 2D we have
            # allmat[0] = D kron I
            # allmat[1] = I kron D
            # and in 3D
            # allmat[0] = D kron I kron I
            # allmat[1] = I kron D kron I
            # allmat[2] = I kron I kron D
            # and so on, for kron meaning the kronecker product.
            
            # For a (n x m) array, the forward difference operator in y-direction (x-direction) (with Neumann/Periodic bc) is a (n*m x n*m) sparse array containing -1,1.
            # Example, for a 3x3 array U, the forward difference operator in y-direction with Neumann bc is a 9x9 sparse array containing -1,1.
            # To create this sparse matrix, we first create a "kernel" matrix, shown below:
            # mat = [-1, 1, 0
            #        0, -1, 1,
            #        0, 0, 0].
            # where the last row is filled with zeros due to the Neumann boundary condition. Then, we use the Kronecker product: allMat[0] = mat x I_m =
            # matrix([[-1., 1., 0., 0., 0., 0., 0., 0., 0.],
            #         [ 0., -1., 1., 0., 0., 0., 0., 0., 0.],
            #         [ 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            #         [ 0., 0., 0., -1., 1., 0., 0., 0., 0.],
            #         [ 0., 0., 0., 0., -1., 1., 0., 0., 0.],
            #         [ 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            #         [ 0., 0., 0., 0., 0., 0., -1., 1., 0.],
            #         [ 0., 0., 0., 0., 0., 0., 0., -1., 1.],
            #         [ 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
            # where I_m is an (mxm) sparse array with ones in the diagonal.
            # Then allmat can be applied to a (3x3) array "flatten" columnwise
            # and represent the forward differences in y direction,i.e.,
            # [U_{21} - U_{11},
            #  U_{31} - U_{21},
            #        0        ,
            #       ...       ,
            #       ...       ,
            #       ...       ,
            #       ...       ,
            #       ...       ,
            #       ...       ]
            # For the x-direction, we have allmat[1] = I_n x mat.

            # For more details, see "Infimal Convolution Regularizations with Discrete l1-type Functionals, S. Setzer, G. Steidl and T. Teuber"
        
            # According to the direction, tmpGrad is either a kernel matrix or sparse eye array, which is updated 
            # using the kronecker product to derive the sparse matrices.
            if i==0:
                tmpGrad = mat
            else: 
                tmpGrad = sp.eye(shape[0])

            for j in range(1, len_shape):

                if j == i:
                    tmpGrad = sp.kron(mat, tmpGrad ) 
                else:
                    tmpGrad = sp.kron(sp.eye(shape[j]), tmpGrad )

            allMat[i] = tmpGrad

    else:
        raise NotImplementedError    

    return allMat        


# solution
u_cvx = cvxpy.Variable(noisy_data.shape)

# fidelity term
fidelity = 0.5 * cvxpy.sum_squares(u_cvx - noisy_data.array)   
regulariser = alpha * tv_cvxpy_regulariser(u_cvx)

# objective
obj =  cvxpy.Minimize( regulariser +  fidelity)
prob = cvxpy.Problem(obj, constraints = [])

# Choose solver ( SCS, MOSEK(license needed) )
tv_cvxpy = prob.solve(verbose = True, solver = cvxpy.SCS)


import numpy as np
tv_cil = pdhg.solution[0]
# compare solution
#%%
show2D([tv_cil, u_cvx.value], title=['CIL', 'CVXPY'])
#%%
np.testing.assert_allclose(tv_cil.array, u_cvx.value,atol=1e-3)  

# %%
