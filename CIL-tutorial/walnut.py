import numpy as np
import os

from cil.io import TXRMDataReader, TIFFWriter
from cil.processors import TransmissionAbsorptionConverter, CentreOfRotationCorrector, Slicer
from cil.plugins.tigre import FBP
from cil.utilities.display import show2D, show_geometry

base_dir = os.path.abspath(r"C:\Users\ofn77899\Data\walnut")
data_name = "valnut"
filename = os.path.join(base_dir, data_name, "valnut_2014-03-21_643_28/tomo-A/valnut_tomo-A.txrm")

data3D = TXRMDataReader(file_name=filename).read()
data = data3D.get_slice(vertical='centre') 

show_geometry(data.geometry)

data = TransmissionAbsorptionConverter()(data)

data.reorder(order='tigre')

ig = data.geometry.get_ImageGeometry()

fbp =  FBP(ig, data.geometry)
recon = fbp(data)

show2D([data, recon], fix_range=(-0.01,0.06))