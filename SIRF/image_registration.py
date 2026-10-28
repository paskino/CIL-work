#%%
import SimpleITK as sitk
import numpy as np
import pysnooper
from scipy.spatial.transform import Rotation, RigidTransform
from math import pi
import matplotlib.pyplot as plt

@pysnooper.snoop()
def register_3d_numpy_arrays(fixed_array: np.ndarray, moving_array: np.ndarray):
    """
    Registers two 3D numpy arrays of the same modality using SimpleITK.
    
    Args:
        fixed_array: 3D numpy array representing the fixed image.
        moving_array: 3D numpy array representing the moving image.
        
    Returns:
        rotation_matrix: 3x3 numpy array
        translation: 3-element numpy array
        registered_array: 3D numpy array of the moving image resampled to match the fixed image
    """
    # Convert numpy arrays to SimpleITK images
    # SimpleITK expects images in (z, y, x) order if coming from numpy, 
    # GetImageFromArray handles this automatically.
    fixed_image = sitk.GetImageFromArray(fixed_array)
    moving_image = sitk.GetImageFromArray(moving_array)
    
    # Cast to float32 for registration
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    # Initialize the registration method
    registration_method = sitk.ImageRegistrationMethod()
    # Similarity metric: Mean Squares is standard for same modality. 
    # Normalized Cross Correlation (NCC) is also a good option.
    registration_method.SetMetricAsMeanSquares()
    
    # Optimizer: Regular Step Gradient Descent
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, 
        minStep=1e-4, 
        numberOfIterations=100, 
        gradientMagnitudeTolerance=1e-8
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    # Setup the initial transform as a rigid 3D transform (Euler3DTransform)
    # CenteredTransformInitializer aligns the centers of the images to start
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, 
        moving_image, 
        sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=True)
    # Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    # Execute the registration
    final_transform = registration_method.Execute(fixed_image, moving_image)
    
    # Resample the moving image using the final transform
    resampled_moving_image = sitk.Resample(
        moving_image, 
        fixed_image, 
        final_transform, 
        sitk.sitkLinear, 
        0.0, 
        moving_image.GetPixelID()
    )
    # Convert resampled image back to numpy array
    registered_array = sitk.GetArrayFromImage(resampled_moving_image)
    
    # Cast back to Euler3DTransform to extract matrix and translation easily
    rigid_transform = sitk.Euler3DTransform(final_transform)
    
    # Get the rotation matrix (returned as a 1D tuple of length 9)
    rotation_matrix_1d = rigid_transform.GetMatrix()
    rotation_matrix = np.array(rotation_matrix_1d).reshape(3, 3)
    
    # Get the translation
    translation = np.array(rigid_transform.GetTranslation())
    
    # SimpleITK transforms are defined relative to a center of rotation.
    # The actual spatial transformation applied to a point p is: R * (p - c) + c + t.
    # If you need the pure translation T such that p' = R*p + T, you can compute it:
    center = np.array(rigid_transform.GetCenter())
    pure_translation = translation + center - np.dot(rotation_matrix, center)
    
    return rotation_matrix, translation, registered_array

def print_rot_matrix(rot):
    print(np.array_str(rot.as_matrix(), precision=3, suppress_small=True))

#%%
# Create synthetic data for demonstration
print("Generating synthetic 3D data...")
fixed_data = np.zeros((50, 50, 50), dtype=np.float32)
fixed_data[15:35, 15:35, 15:35] = 1.0  # A cube in the center
fixed_data += np.random.normal(0, 0.1, fixed_data.shape).astype(np.float32)

# Create moving data by translating the cube slightly
moving_data = np.zeros((50, 50, 50), dtype=np.float32)
moving_data[15:35, 15:35, 15:35] = 1.0  
moving_data += np.random.normal(0, 0.3, moving_data.shape).astype(np.float32)
#%%
order = 'ZXY' # <--

# reordered_euler_angles = (15/180*np.pi, 0, 0)  # <--
# scipy_matrix = Rotation.from_euler(order, reordered_euler_angles).as_matrix()
zangle = 1 * 15/180*np.pi
xshift = 1 * 10
rot = Rotation.from_euler('Z', zangle, degrees=False)

print_rot_matrix(rot)

t = RigidTransform.from_translation((xshift,0,0))
from scipy.ndimage import affine_transform

center = np.array(moving_data.shape) / 2
center_shift = RigidTransform.from_translation(center)
center_unshift = RigidTransform.from_translation(-center)
tt = center_shift * rot * t * center_unshift
scipy_matrix = tt.as_matrix()

# print(np.array_str(scipy_matrix, precision=3, suppress_small=True))
print_rot_matrix(tt)

rotated_moving_data = affine_transform(
    moving_data,
    matrix=scipy_matrix,
    order=1,          # 1 = linear interpolation, 3 = cubic interpolation
    mode='constant',  # How to handle pixels outside the bounds
    cval=0.0          # Value for pixels outside the bounds
)

#%%
plt.figure()
plt.subplot(1,3,2)
plt.imshow(moving_data[:,:,25], cmap='gray')
plt.title("Moving Data (Slice 25)")

plt.subplot(1,3,1)
plt.imshow(fixed_data[:,:,25], cmap='gray')
plt.title("Fixed Data (Slice 25)")

plt.subplot(1,3,3)
plt.imshow(rotated_moving_data[:,:,25] - moving_data[:,:,25], cmap='gray')
plt.title("Difference (Slice 25)")
plt.show()

#%%
print("Registering...")
rot_mat, trans, registered_array = register_3d_numpy_arrays(fixed_data, rotated_moving_data)

#%%
print("Registration completed.")
print("Rotation Matrix:")
# SciPy's affine_transform is doing a backward mapping, if you hand it a matrix 
# M, it is effectively applying the inverse mapping to create the rotated image.
# Since your matrix is a pure rotation matrix, its inverse is equal to its transpose.

print(rot_mat[::-1,::-1].T)
print("Translation:")
print(trans)

print_rot_matrix(Rotation.from_matrix(rot.as_matrix()))
print_rot_matrix(RigidTransform.from_matrix(t.as_matrix()))
# %%

plt.figure()
plt.subplot(1,3,1)
plt.imshow(fixed_data[:,:,25], cmap='gray')
plt.title("Fixed Data (Slice 25)")

plt.subplot(1,3,2)
plt.imshow(rotated_moving_data[:,:,25], cmap='gray')
plt.title("Moving Data (Slice 25)")

plt.subplot(1,3,3)
plt.imshow(registered_array[:,:,25], cmap='gray')
plt.title("Registered Data (Slice 25)")

plt.show()

# %%
