#%%
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import matplotlib.patches as patches
from cil.utilities.display import show2D
import os

dirname = os.path.abspath(r"C:\Users\ofn77899\Dev\CIL-work\MCIR_cardiorespiratory\spdhg_mcir_results")

recons = {}
# [80, 200]
algo = ['PDHG', 'SPDHG']
epochs = [80, 200]
for a in algo:
    for e in epochs:
        recons[a + '_' + str(e)+'_resp'] = np.load(os.path.join(dirname, 'Fig_Invivo_Recon_resp' + a + '_' + str(e) + '_epochs.npy'))
# Fig_Invivo_Recon_respPDHG_80_epochs
# Fig_Invivo_Recon_respSPDHG_80_epochs

# [20 200]
epochs = [20, 200]
# Fig_Invivo_ReconPDHG_20_epochs
# Fig_Invivo_ReconSPDHG_20_epochs
for a in algo:
    for e in epochs:
        recons[a + '_' + str(e)] = np.load(os.path.join(dirname, 'Fig_Invivo_Recon' + a + '_' + str(e) + '_epochs.npy'))

# no MCIR
# Fig_Invivo_ReconPDHG_680_epochs_wo_MCIR.npy
# Fig_Invivo_Recon_respPDHG_680_epochs_wo_MCIR.npy
recons['PDHG_680_noMCIR'] = np.load(os.path.join(dirname, 'Fig_Invivo_ReconPDHG_680_epochs_wo_MCIR.npy'))
recons['PDHG_680_resp_noMCIR'] = np.load(os.path.join(dirname, 'Fig_Invivo_Recon_respPDHG_680_epochs_wo_MCIR.npy'))

#%%
from cil.utilities.display import show2D
import matplotlib.pylab as pylab
params = {'axes.titlesize':'24'}
pylab.rcParams.update(params)

side = 70
x1, y1 = 35, 65
x2, y2 = x1 + side, y1 + side

#%% Choose which ones you want to show:
# respiratory only
which = ['PDHG_80_resp', 'SPDHG_80_resp', 'PDHG_200_resp', 'SPDHG_200_resp',]
titles = ['Respiratory PDHG 80 epochs',  'Respiratory SPDHG 80 epochs',\
          'Respiratory PDHG 200 epochs', 'Respiratory SPDHG 200 epochs']

# cardio-respiratory
which = ['PDHG_20', 'SPDHG_20', 'PDHG_200', 'SPDHG_200',]
titles = ['Cardio-Respiratory PDHG 20 epochs',  'Cardio-Respiratory SPDHG 20 epochs',\
          'Cardio-Respiratory PDHG 200 epochs', 'Cardio-Respiratory SPDHG 200 epochs']


# shows inset
show2D([recons[el][y1:y2, x1:x2] for el in which], \
     origin='upper-left', cmap='gray', num_cols=2, title=titles)

#%% 

# set fontszie xticks/yticks
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15

fig = plt.figure(figsize=(20, 20))

grid = AxesGrid(fig, 111,
                nrows_ncols=(2, 2),
                axes_pad=(0.08, 0.4),
                cbar_mode='single',
                cbar_location='bottom',
                cbar_size = 0.5,
                cbar_pad=0.1
                )

k = 0

alg = 'SPDHG'

for i,el in enumerate(which):
    alg, e = el.split('_')
    ax = grid[i] 
    data = recons[alg + '_' + str(e)]
    im = ax.imshow(data, cmap="gray")
    
    
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='w', facecolor='none')
    ax.add_patch(rect)   
    
    ax.set_title(titles[i].split("Cardio-Respiratory")[1], fontsize=25)


    ax.set_xticks([])
    ax.set_yticks([]) 
   
cbar = ax.cax.colorbar(im)
# cbar.ax.set_xlabel('Attenuation', fontsize=25)
fig.set_tight_layout(True)
plt.show()
# %%
