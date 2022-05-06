"""SPDHG reconstruction with TV.
Usage:
  QM_SPDHG_explicit.py [--help | options]
Options:
  --alpha=<val>   regularisation parameter
  --num_phys_subs=<int> number of physical subsets
                        [default: 20]
  --adaptive-georg  use adaptive sigma and tau with Georg Schramm's rule

"""

#%% imports
from cil.framework import AcquisitionGeometry
from cil.optimisation.functions.MixedL21Norm import MixedL21Norm
from cil.optimisation.operators import GradientOperator, BlockOperator, IdentityOperator
from cil.optimisation.functions import BlockFunction, IndicatorBox
from cil.optimisation.functions import TotalVariation
from cil.io import NEXUSDataWriter
from cil.recon import FDK

from cil.plugins.tigre import ProjectionOperator as POT
# from cil.plugins.astra import ProjectionOperator as POA


# from cil.plugins.ccpi_regularisation.functions.regularisers import FGP_TV
# from cil.utilities.jupyter import islicer
from cil.utilities.display import show2D
from cil.processors import Slicer

from cil.optimisation.algorithms import FISTA, PDHG, SPDHG
from cil.optimisation.functions import LeastSquares, ZeroFunction, L2NormSquared, L1Norm
import matplotlib.pyplot as plt

import numpy as np
import os
import math

import docopt

__version__ = '0.1.0'
args = docopt.docopt(__doc__, version=__version__)

#%%
class AcquisitionGeometrySubsetGenerator(object):
    '''AcquisitionGeometrySubsetGenerator is a factory that helps generating subsets of AcquisitionData
    
    AcquisitionGeometrySubsetGenerator generates the indices to slice the data array in AcquisitionData along the 
    angle dimension with 4 methods:

    1. random: picks randomly between all the angles. Subset may contain same projection as other
    2. random_permutation: generates a number of subset by a random permutation of the indices, thereby no subset contain the same data.
    3. uniform: divides the angles in uniform subsets without permutation
    4. stagger: generates number_of_subsets by interleaving them, e.g. generating 2 subsets from [0,1,2,3] would lead to [0,2] and [1,3]

    The factory is not to be used directly rather from the AcquisitionGeometry class.

    '''
    
    ### Changes in the Operator required to work as OS operator
    @staticmethod
    def generate_subset(acquisition_data, number_of_subsets, method='random'):
        
        ag = acquisition_data.geometry
        angles = ag.angles.copy()
        if method == 'random':
            indices = [ AcquisitionGeometrySubsetGenerator.random_indices(angles, number_of_subsets) 
              for _ in range(number_of_subsets) ] 
            
        elif method == 'random_permutation':
            rndidx = np.asarray(range(len(angles)))
            np.random.shuffle(rndidx)
            indices = AcquisitionGeometrySubsetGenerator.uniform_groups_indices(rndidx, number_of_subsets)
            
        elif method == 'uniform':
            rndidx = np.asarray(range(len(angles)))
            indices = AcquisitionGeometrySubsetGenerator.uniform_groups_indices(rndidx, number_of_subsets)
            
        elif method == 'stagger':
            idx = np.asarray(range(len(angles)))
            indices = AcquisitionGeometrySubsetGenerator.staggered_indices(idx, number_of_subsets)
        else:
            raise ValueError('Can only do {}. got {}'.format(['random', 'random_permutation', 'uniform'], method))
        
        # return indices    
        subsets = [] 
        for idx in indices:
            g = ag.copy()
            angles = ag.angles[idx]

            #preserve initial angle and unit
            g.config.angles.angle_data = angles
            data = g.allocate(0)
            data.fill(acquisition_data.as_array()[idx])
            subsets.append( data )
        return subsets
    
    @staticmethod
    def uniform_groups_indices(idx, number_of_subsets):
        indices = []
        groups = int(len(idx)/number_of_subsets)
        for i in range(number_of_subsets):
            ret = np.asarray(np.zeros_like(idx), dtype=bool)
            for j,el in enumerate(idx[i*groups:(i+1)*groups]):
                ret[el] = True
                
            indices.append(ret)
        return indices
    @staticmethod
    def random_indices(angles, number_of_subsets):
        N = int(np.floor(float(len(angles))/float(number_of_subsets)))
        indices = np.asarray(range(len(angles)))
        np.random.shuffle(indices)
        indices = indices[:N]
        ret = np.asarray(np.zeros_like(angles), dtype=bool)
        for i,el in enumerate(indices):
            ret[el] = True
        return ret
    @staticmethod
    def staggered_indices(idx, number_of_subsets):
        indices = []
        # groups = int(len(idx)/number_of_subsets)
        for i in range(number_of_subsets):
            ret = np.asarray(np.zeros_like(idx), dtype=bool)
            indices.append(ret)
        i = 0
        while i < len(idx):    
            for ret in indices:
                ret[i] = True
                i += 1
                if i >= len(idx):
                    break
                
        return indices
    @staticmethod
    def get_new_indices(index):
        newidx = []
        for idx in index:
            ai = np.where(idx == True)[0]
            for i in ai:
                newidx.append(i)
        return np.asarray(newidx)


#%%
# calculate the objective only on the first subset (for speed)
def update_objective(self):
    # p1 = self.f(self.operator.direct(self.x)) + self.g(self.x)
    p1 = 0.
    for i,op in enumerate(self.operator.operators):
        if i > 1:
            break
        p1 += self.f[i](op.direct(self.x))
    p1 += self.g(self.x)

    d1 = - self.f.convex_conjugate(self.y_old)
    tmp = self.operator.adjoint(self.y_old)
    tmp *= -1
    d1 -= self.g.convex_conjugate(tmp)

    self.loss.append([p1, d1, p1-d1])



#%% parser
def parse_QMUL_geometry(filename):
    qmul = {}
    qmul['CENTRE'] = []
    with open(filename,'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            field, value = line.split(':')
            # strip all leading empty spaces
            value = value.strip()
            try:
                v, unit = value.split(' ')
            except ValueError as ve:
                v = value
                unit = None
            try:
                vv = float(v)
                if vv % 1 == 0:
                    vv = int(vv) 
                if unit is not None and unit == 'u':
                    vv = vv * 1e-4 # convert to cm
            except ValueError as ve:
                vv = value

            try:
                qmul[field].append(vv)
            except:
                qmul[field]  = vv

        if qmul['NUMBER OF BLOCKS'] == 1 or len(qmul['CENTRE']) == 1:
            qmul['CENTRE'] = qmul['CENTRE'][-1]
    return qmul


from cil.optimisation.functions import Function
class ScaledArgFunction(Function):
    def __init__(self, function, scalar):
        self.function = function
        self.scalar = scalar

    def __call__(self, x):
        x *= self.scalar
        ret = self.function(x)
        x /= self.scalar
        return ret

    def gradient(self, x, out=None):
        # fix the gradient
        x *= self.scalar

        return self.scalar * self.function.gradient(x, out=out)

    def proximal(self, x, tau, out=None):
        # eq 6.6 of https://archive.siam.org/books/mo25/mo25_ch6.pdf
        should_return = False
        if out is None:
            out = x * 0
            should_return = True
        x *= self.scalar
        self.function.proximal( x, tau * self.scalar**2, out=out)
        x /= self.scalar
        out /= self.scalar
        if should_return:
            return out
    def convex_conjugate(self, x):
        # https://en.wikipedia.org/wiki/Convex_conjugate#Table_of_selected_convex_conjugates
        x /= self.scalar
        ret = self.function.convex_conjugate(x)
        x *= self.scalar
        return ret

if __name__ == '__main__':
    # #%% parse input file
    directory = os.path.abspath("/mnt/data/QMUL/")
    # tfile = open('title.txt', 'r')
    # fname = tfile.readline()
    # tfile.close()
    # title = fname.strip()
    #title = 'yj_lltm_an_pk_18'
    title = 'yj_LRSM_PK15'
    fname = os.path.join(directory, title + '.cra')
    #fname = os.path.join(directory,'phantooth.cra')

    meta_data = parse_QMUL_geometry(fname)
    print(meta_data)

#%% set-up aquisition geometry
    angles = np.linspace(0,360,meta_data['NUMBER OF PROJECTIONS'],endpoint=False)
    initial_angle=270-meta_data['ROTATE ANGLE']

    if 'CLOCKWISE' not in meta_data.keys():
        angles *= -1
        initial_angle *= -1

    ag = AcquisitionGeometry.create_Cone3D(source_position=(0,-meta_data['DISTANCE'],0), detector_position=(0,0,0))\
                .set_panel((meta_data['DETECTOR X PIXELS'],meta_data['DETECTOR Z PIXELS']), meta_data['SAMPLE SPACING'])\
                .set_angles(angles, initial_angle)\
                .set_labels(['angle','vertical','horizontal'])

    print(ag)



#%% set-up image geometry
    ig = ag.get_ImageGeometry()

    distance = meta_data['DISTANCE']/meta_data['SAMPLE SPACING']
    phi = math.atan2(meta_data['DETECTOR X PIXELS']/2.0, distance)
    max_XY = (int(distance*math.sin(phi)+1.0))*2
    if 'X DIMENSION' in meta_data.keys():
        XYdim = meta_data['X DIMENSION']
    else:
        XYdim = max_XY
    if XYdim > max_XY:
        XYdim = max_XY

    Zdim = meta_data['DETECTOR Z PIXELS']
    maxZ = (int((distance - XYdim/2)*Zdim/distance/2))*2
    if 'Z DIMENSION' in meta_data.keys():
        Zdim = meta_data['Z DIMENSION']
    else:
        Zdim = maxZ
    if Zdim > maxZ:
        Zdim = maxZ

    scale = 1.0
    if 'SCALE' in meta_data.keys():

        scale = meta_data['SCALE']
    ig.voxel_size_x /= scale
    ig.voxel_size_y /= scale
    ig.voxel_size_z /= scale
    XYdim = int(XYdim*scale)
    XYdim -= XYdim % 2
    Zdim = int(Zdim*scale)
    Zdim -= Zdim % 2
    ig.voxel_num_x = XYdim
    ig.voxel_num_y = XYdim
    # ig.voxel_num_z = 50 #Zdim for all
    ig.voxel_num_z = Zdim

    print(ig)
    sizfile = open(title+'.siz', 'w')
    sizfile.write(str(XYdim) + ' ' + str(XYdim) + ' ' + str(Zdim*meta_data['NUMBER OF BLOCKS']) + ' 0\n')
    sizfile.close()


    
#%% read in data
    path_in = os.path.join(directory, meta_data['FILE NAME'])
    ad = ag.allocate(0)

    #reconstructor set up

#%% pick rows

    # ad.fill(0)
    data_ones = ig.allocate(1)
    PO = POT(ig, ad.geometry)
    fp = PO.direct(data_ones)

    tmp = fp.as_array()
    tmp = tmp.mean(axis=0)
    data_ind = np.nonzero(tmp)

    row_start_ind = data_ind[0].min()
    row_end_ind = data_ind[0].max()

    #santity check
    if ad.shape[1] -1 - row_start_ind != row_end_ind:
        print("check shape")


#%% read in single block data to compare

    block_num=2

    block_offset = block_num * ad.size*4
    ad.fill(np.fromfile(path_in, dtype=np.float32, count=ad.size, sep='',offset=block_offset).reshape(ag.shape))

    #set centre of rotation for block
    try:
        centre_pix = meta_data['CENTRE'][block_num]
    except:
        centre_pix = meta_data['CENTRE']

    obj_offset_x = (centre_pix - (meta_data['DETECTOR X PIXELS']-1)/2) * meta_data['SAMPLE SPACING']

    ad.geometry.config.system.rotation_axis.position = (obj_offset_x, 0, 0)

#%%
    if row_start_ind == 0:
        data = ad
    else:
        roi = {'vertical':(row_start_ind,-row_start_ind)}
        slicer = Slicer(roi=roi)
        ad_crop = slicer(ad)
        data = ad_crop


    #reco = FDK_reco[block_num]

    initial = ig.allocate(0)
#%%

    alpha = args['--alpha']
    num_subsets = args['--num_phys_subs']
    subsets = AcquisitionGeometrySubsetGenerator.generate_subset(data, num_subsets, method='random_permutation') 

    operators = []
    functions = []
    for i,sub in enumerate(subsets):
        A1 = POT(ig, sub.geometry)
        print ("Calculate norm for subset ", i)
        if i == 0:
            if num_subsets == 25:
                A1._norm = 0.38119105
            norm = A1.norm(iterations=5)
        else:
            if len(sub.geometry.config.angles.angle_data) == \
                len(subsets[i-1].geometry.config.angles.angle_data):
                print ("can use pre calculated norm")
                A1._norm = norm
            else:
                print ("recalculate norm for subset ", i)
                norm = A1.norm(iterations=5)
        operators.append( A1 )
        functions.append(
            L2NormSquared(b=sub)
        )
        
        # functions.append( ScaledArgFunction( L2NormSquared(b=sub), operators[i].norm()) )
    physical_subsets_prob = 1./len(subsets)

    g1 = GradientOperator(ig)
    g1._norm = 2282.4083771314895

    if args['--rescale-operators']:
        ops = [ 1/el.norm() * el for el in operators]
        operators = ops
        operators.append( (1 / g1.norm(iterations=5)) * g1 )
        # functions.append( ScaledArgFunction( MixedL21Norm(), g1.norm()) )
        functions.append(  
            ( alpha / norm**2 ) * MixedL21Norm()
        )
    else:
        operators.append( g1 )
        # functions.append( ScaledArgFunction( MixedL21Norm(), g1.norm()) )
        functions.append(  
            ( alpha ) * MixedL21Norm()
        )

    K = BlockOperator(*operators)
    f = BlockFunction(*functions)
    g = IndicatorBox(lower=0)

    #%%

    def get_probs(gamma, physical_subsets_prob, subsets):
        '''Get the probabilities for SPDHG
        
        gamma : float between 0 and 1. 0 means 0 probability of physical subsets'''
        prob = [gamma * physical_subsets_prob for _ in subsets]
        prob.append(1 - gamma)
        return prob

    prob = get_probs(0.75, physical_subsets_prob, subsets)

    

    recon_dir = os.path.abspath('/mnt/data/edo/Dev/recon/SPDHG/tooth-norm')
    algo = SPDHG(f=f,g=g,operator=K ,max_iteration=1e6,
                prob=prob,
                log_file=os.path.join(recon_dir, 'numba_full_volume_alpha_{}.log'.format(alpha)))



    algo.update_objective_interval = num_subsets+1#num_subsets+1
    num_epochs=10
    #%%
    # nvidia-smi dmon -s uv
    # run TV step
    # algo.prob = prob = get_probs(0, physical_subsets_prob, subsets)
    # algo.run(10, print_interval=1)
    # print(algo.timing)
    # # run projection step
    # algo.prob = prob = get_probs(1, physical_subsets_prob, subsets)
    # algo.run(10, print_interval=1)
    # print(algo.timing)

    print ("num_subsets", num_subsets)
    #%%
    ndual_subsets = num_subsets + 1
    for i in range(10):
        # print("Current Time =", datetime.now().strftime("%H:%M:%S"))
        # print("iteration block {0} - {1}".format(i*num_epochs,(i+1)*num_epochs-1))
        algo.run(num_epochs*(num_subsets+1), verbose=2)
        if args['--adaptive']:
            gamma = algo.solution.norm()
            sigma = [ gamma / (K.norm()) for _ in range(ndual_subsets)]
            tau = 1 / (gamma * K.norm())
            algo.sigma = sigma
            algo.tau = tau


        path_out = os.path.join(recon_dir, 'numba_normSPDHG_eTV_alpha_{0}_it_{1}'.format(alpha, algo.iteration))
        print ("Writing to ", path_out)
        writer = NEXUSDataWriter(file_name=path_out,data=algo.solution/(K.norm()**2),compression=0)
        writer.write()

        # path_out = os.path.join(recon_dir, 'normSPDHG_eTV_alpha_{0}_it_{1}.png'.format(alpha, algo.iteration))
        # plots = [algo.solution/(K.norm()**2)]*3
        # ranges = [(-0.5,3.5),(2.5,3.2),(1.5,2.1)]
        # show2D(plots,fix_range=ranges,num_cols=1).save(path_out)


    print ("K norm ", K.norm())
    writer = NEXUSDataWriter(file_name=path_out,data=algo.solution/(K.norm()**2),compression=0)
    writer.write()
    #K norm  4.58257569495584
    #%%
    print('fin')