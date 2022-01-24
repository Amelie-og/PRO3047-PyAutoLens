"""
Visualize datasets and plot their components (image, PSF, Noise-Map, etc).
Use this to investigate features of an image before fitting it.
(Determine size of the mask to apply, etc.)
"""

# Import packages
from os import path
import autolens as al
import autolens.plot as aplt

# Load and resize the image of the dataset
dataset_name = "B1938-ER"
dataset_path = path.join("dataset", "CASTLES", dataset_name)
image = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "image.fits"),
    pixel_scales=0.1
)
image_processed = al.preprocess.array_with_new_shape(
    array=image,
    new_shape=(150, 150)
)

# Load and resize a noise_map
dataset_path_1 = path.join("dataset", "imaging", "preprocess", "imaging")
noise_map = al.Array2D.from_fits(
    file_path=path.join(dataset_path_1, "noise_map.fits"),
    pixel_scales=0.1
)
noise_map_processed = al.preprocess.array_with_new_shape(
    array=noise_map,
    new_shape=(150, 150)
)
# Create a new noise_map based on an empty part of a picture from the same experiment
# Load an image from the same detector (here: WPFC2)
dataset_path = path.join("dataset", "CASTLES", "B0218")
image_empty = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "image.fits"),
    pixel_scales=0.1
)
# Run over the values of the noise_map and assign value (create the 'tiling')
for i in range(150):
    if i > 147: w = i - 147
    elif i > 98: w = i - 98
    elif i > 49: w = i - 49
    else: w = i
    for j in range(150):
        if j > 147: x = j - 147
        elif j > 98: x = j - 98
        elif j > 49: x = j - 49
        else: x = j
        noise_map_processed.slim[i*150+j] = image_empty.slim[w*256+x]
        print("For ixX = ", i, "x", x, ", ORIGIN:", image_empty.slim[w*256+x])
        print("For ixj = ", i, "x", j, ", RESULT:", noise_map_processed.slim[i*150+j])

# Load (and resize) a psf
dataset_path_2 = path.join("dataset", "imaging", "no_lens_light", "mass_sie__source_sersic")
# can also load a different one by using dataset_path_1 (both are already in 21x21)
psf = al.Kernel2D.from_fits(
    file_path=path.join(dataset_path_2, "psf.fits"),
    hdu=0,
    pixel_scales=0.1
)
# psf_processed = al.preprocess.array_with_new_shape(array=psf, new_shape=(21, 21))

# Create an 'Imaging' object with the image, noise_map and psf
imaging = al.Imaging(image=image_processed,
                     noise_map=noise_map_processed,
                     psf=psf)
# Plot dataset (Image, Noise-Map, Point Spread Function, Signal-To-Noise Map, Inversion Noise-Map,
# Potential Chi-Squared Map)
imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

# --- Test different masks before determining which one will be used for the modeling) ---
# Create and apply a mask 
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, 
    pixel_scales=imaging.pixel_scales, 
    radius=4.0)
imaging = imaging.apply_mask(mask=mask)

# Plot the imaging dataset with the mask to determine if it is a good fit
imaging_plotter = aplt.ImagingPlotter(
    imaging=imaging, 
    visuals_2d=aplt.Visuals2D(mask=mask))
imaging_plotter.subplot_imaging()