#%%
from cil.utilities.dataexample import SIMULATED_CONE_BEAM_DATA, SIMULATED_PARALLEL_BEAM_DATA, SIMULATED_SPHERE_VOLUME
from cil.utilities.display import show2D
from cil.plugins.tigre import ProjectionOperator, FBP as FBP_t
from cil.recon import FDK as FBP

from cil.utilities.display import show2D
from cil.processors import CentreOfRotationCorrector
import numpy as np

#%%
image_data = SIMULATED_SPHERE_VOLUME.get()
data = SIMULATED_CONE_BEAM_DATA.get()
data=np.log(data)
data*=-1.0
data.reorder('tigre')

#%%
data_offset = data.copy()
reco1 = FBP(data_offset).run()
data_offset.geometry.config.system.rotation_axis.position=[20,0,0]
reco2 = FBP(data_offset).run()

show2D([reco1,reco2],title=['original','offset data'])
#%%
data_offset = data.copy()
reco1 = FBP(data_offset).run()
data_offset.geometry.config.system.rotation_axis.position=[20,0,0]
data_offset.geometry.config.
reco2 = FBP(data_offset).run()

show2D([reco1,reco2],title=['original','offset data'])
#%%


corrector = CentreOfRotationCorrector.image_sharpness('centre',FBP_t)
corrector.set_input(data_offset)
data_centred = corrector.get_output()

#%%
#method = 'interpolated'
method = 'Siddon'

#%%
ag = data_offset.geometry
ig = ag.get_ImageGeometry()
PO = ProjectionOperator(ig, ag, direct_method=method)
reco = FBP(data_offset).run()

bp = PO.adjoint(data_offset)
fp = PO.direct(bp)
show2D([reco,bp,fp])

#%%
ag = data_centred.geometry
ig = ag.get_ImageGeometry()
PO = ProjectionOperator(ig, ag, direct_method=method)
reco = FBP(data_centred).run()


bp = PO.adjoint(data_centred)
fp = PO.direct(bp)
show2D([reco,bp,fp])
# %%
