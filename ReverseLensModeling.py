from os import path
import autolens as al
import autolens.plot as aplt

### Amelies Code to resize the images
# Load and resize the image of the dataset
dataset_name = "B1938"
dataset_path = path.join("dataset", "Castles",  dataset_name)
image = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "B1938.fits"),
    pixel_scales=0.1    ##units still arbitrary
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


dataset_path = path.join("dataset", "Castles", "B1938")
image_empty = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "B1938.fits"),
    pixel_scales=0.1
)


# Load (and resize) a psf
dataset_path_2 = path.join("dataset", "imaging", "no_lens_light", "mass_sie__source_sersic")
# can also load a different one by using dataset_path_1 (both are already in 21x21)
psf = al.Kernel2D.from_fits(
    file_path=path.join(dataset_path_2, "psf.fits"),
    hdu=0,
    pixel_scales=0.1
)

imaging = al.Imaging(image=image_processed,
                     noise_map=noise_map_processed,
                     psf=psf
)

###Chapter4T3
##create mask around lens
mask = al.Mask2D.circular_annular(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    inner_radius=0.65,
    outer_radius=2.2,
    origin=(0.25, 0.0),
)

visuals_2d = aplt.Visuals2D(mask=mask)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging, visuals_2d=visuals_2d)
imaging_plotter.figures_2d(image=True)

imaging = imaging.apply_mask(mask=mask)

#load the lens data

lens_galaxy = al.Galaxy(
    redshift=0.844,
    mass=al.mp.SphIsothermal(
        centre=(0.25, 0.0),
        einstein_radius=2.0,
    ),
    shear=al.mp.ExternalShear(elliptical_comps=(0.05, 0.05)),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, al.Galaxy(redshift=1.0)])

source_plane_grid = tracer.traced_grids_of_planes_from(grid=imaging.grid)[1]

rectangular = al.pix.Rectangular(shape=(25, 25))

mapper = rectangular.mapper_from(grid=source_plane_grid)

include_2d = aplt.Include2D(mask=True, mapper_source_grid_slim=True)

mapper_plotter = aplt.MapperPlotter(mapper=mapper, include_2d=include_2d)
mapper_plotter.subplot_image_and_mapper(image=imaging.image),

inversion = al.Inversion(
    dataset=imaging,
    mapper_list=[mapper],
    regularization_list=[al.reg.Constant(coefficient=1.0)],
)

print(inversion.reconstruction)
print(inversion.mapped_reconstructed_image)

include_2d = aplt.Include2D(mask=True)

inversion_plotter = aplt.InversionPlotter(inversion=inversion, include_2d=include_2d)
inversion_plotter.figures_2d(reconstructed_image=True)
inversion_plotter.figures_2d_of_mapper(mapper_index=0, reconstruction=True)
