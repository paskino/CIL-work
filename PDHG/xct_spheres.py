#%%
from cil.utilities import dataexample
from cil.utilities.display import show2D
from cil.recon import FDK
from cil.processors import TransmissionAbsorptionConverter, Slicer

ground_truth = dataexample.SIMULATED_SPHERE_VOLUME.get()

data = dataexample.SIMULATED_CONE_BEAM_DATA.get()
twoD = True
if twoD:
    data = data.get_slice(vertical='centre')
    ground_truth = ground_truth.get_slice(vertical='centre')

absorption = TransmissionAbsorptionConverter()(data)
absorption = Slicer(roi={'angle':(0, -1, 5)})(absorption)

ig = ground_truth.geometry

#%%
recon = FDK(absorption, image_geometry=ig).run()
#%%
show2D([ground_truth, recon], title = ['Ground Truth', 'FDK Reconstruction'], origin = 'upper', num_cols = 2)

# %%
from cil.plugins.tigre import ProjectionOperator
from cil.optimisation.algorithms import FISTA 
from cil.optimisation.functions import LeastSquares, IndicatorBox, ZeroFunction, TotalVariation
from cil.optimisation.operators import GradientOperator
from cil.optimisation.utilities import callbacks

#%%
A = ProjectionOperator(image_geometry=ig, 
                       acquisition_geometry=absorption.geometry)

F = LeastSquares(A = A, b = absorption)
G = IndicatorBox(lower=0)

grad = GradientOperator(domain_geometry=ig)
#%%
alpha = 0.1 * A.norm()/grad.norm()
G = alpha * TotalVariation(max_iteration=2)

algo = FISTA(initial = ig.allocate(), f = F, g = G)



#%%
algo.run(100, callbacks=[callbacks.ProgressCallback()])
# %%

show2D([ground_truth, algo.solution], title = ['Ground Truth', 'FISTA Reconstruction'], origin = 'upper', num_cols = 2)
# %%
