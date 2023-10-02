#%%
from cil.io import NEXUSDataReader
from cil.utilities.quality_measures import mse
from cil.framework import ImageData, ImageGeometry
from cil.utilities.display import show2D, show_geometry, show1D
import os
from glob import glob
import json
import numpy as np
from utils import *

savedir = os.path.join(os.getcwd(), 'results')
data_dir = os.path.abspath('/opt/data/ICASSP24/train/train')


bbox = [150,230, 120,200]
vertical_slice = 170
horizontal_y = 180

dose = 'low'
phantom = 0

ig = create_ImageGeometry()
# clean Image
gt = ImageData(np.load(os.path.join(data_dir, f'{phantom:04d}_clean_fdk_256.npy')),\
                         geometry=ig.copy())

fdk = ImageData(np.load(os.path.join(data_dir, f'{phantom:04d}_fdk_{dose}_dose_256.npy')),\
                         geometry=ig.copy())


# 'n_subs': n_subs,
#      'gamma': gamma,
#      'epochs': epochs ,
#      'alpha': alpha,
#      'mse': dmse,
#      'objective': spdhg.objective,
#      'solution': fname2

#%%
# Dependency on gamma
alpha = 60.
results = []
solutions = []
for fname in glob(os.path.join(os.getcwd(), 'results','*.json')):
    with open(fname, 'r') as f:
        run = json.load(f)
    if run['alpha'] == alpha:
        results.append(run)
        solutions.append(NEXUSDataReader(file_name=run['solution']).read())
#%%

# sort by gamma
gammas = np.asarray([el['gamma'] for el in results])
idx = np.argsort(gammas)

show2D([solutions[i].get_slice(vertical=vertical_slice)\
               .as_array()[bbox[0]:bbox[1], bbox[2]:bbox[3]] \
          for i in idx],
          title=['alpha={} gamma {:0.1f}'.format(alpha, results[i]['gamma']) \
               for i in idx], 
          num_cols=len(idx))

#%%
show2D([solutions[i].get_slice(vertical=vertical_slice) \
          for i in idx],
          title=['alpha={} gamma {:0.1f}'.format(alpha, results[i]['gamma']) \
               for i in idx], 
          num_cols=len(idx))

#%%
n_subs = run['n_subs']
gamma = run['gamma']
epochs = run['epochs']
alpha = run['alpha']
dmse = run['mse']
objective = run['objective']
fname2 = run['solution']
solution = NEXUSDataReader(file_name=fname2).read()

#%%

radius = 1
mfdk = fdk.apply_circular_mask(radius=radius, in_place=False)
malgo = solution.apply_circular_mask(radius=radius, in_place=False)
mgt = gt.apply_circular_mask(radius=radius, in_place=False)

amax = max([mfdk.max(), malgo.max(), gt.max()])
amin = min([mfdk.min(), malgo.min(), gt.min()])
# amin = 0
# amax = 2
diff = mgt-malgo
lim = max(abs(diff.max()), abs(diff.min()))

# zoom in
#%%
show2D([el.get_slice(vertical=vertical_slice)\
               .as_array()[bbox[0]:bbox[1], bbox[2]:bbox[3]] \
          for el in [malgo, mfdk, mgt, diff]], \
     cmap=['viridis', 'viridis', 'viridis', 'seismic'],\
     title=[ f'SPDHG MSE {dmse}', 'FDK','Ground Truth', 'Difference'],\
     fix_range=[
          # (amin, amax), (amin, amax), (amin, amax), 
          None, None, None,
          (-lim/2, lim/2)])
# %%
show2D([el.get_slice(vertical=vertical_slice) \
     for el in [malgo, mfdk, mgt, diff]], \
     cmap=['viridis', 'viridis', 'viridis', 'seismic'],\
     title=[ f'SPDHG MSE {dmse}', 'FDK','Ground Truth', 'Difference'],\
     fix_range=[(amin, amax), (amin, amax), (amin, amax), (-lim/2, lim/2)])
# %%

show1D([el.get_slice(vertical=vertical_slice) \
     for el in [malgo, mfdk, mgt]], \
          label = ['SPDHG', 'FDK', 'Ground Truth'],\
          slice_list = [('horizontal_y', horizontal_y)], \
          title=f'Comparison of the reconstruction at y={horizontal_y}')