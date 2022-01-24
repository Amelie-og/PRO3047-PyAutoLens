"""
This script is used to simulate Hubble Space Telescope (HST) imaging of a strong lens.
In real data of a strong lens, there are lots of other effects (than the actual light coming from the profile)
in our strong lens imaging: noise, diffraction due to the telescope optics, etc.).
"""

from os import path
import autolens as al
import autolens.plot as aplt

# Create a 2D grid (to make the strong lens using a tracer)
grid = al.Grid2D.uniform(shape_native=(150, 150), pixel_scales=0.1)

# Create the profiles (= model) that will be simulated and pass them to the tracer
# TODO: Change this to create various strong lens images
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllDevVaucouleurs(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.5),
        intensity=1.0,
        effective_radius=1.6,
        mass_to_light_ratio=1.0, ),
)
source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SphSersic(
        centre=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=1.0, ),
)
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

# Plot the tracer's image to visualize the image that will be simulated
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.figures_2d(image=True)

# --- Simulation of the Hubble Space Telescope (HST) imaging ---

# Create the Point-Spread Function (PSF) (= models how the light is diffracted/blurred as it enters the telescope)
# PSF is represented using a 'Kernel2D' object (here: 2D Gaussian)
psf = al.Kernel2D.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

# Create the image that will be used for the simulation (padded with zeros around its edge - avoid edge-effects)
normal_image = tracer.image_2d_from(grid=grid)
padded_image = tracer.padded_image_2d_from(grid=grid, psf_shape_2d=psf.shape_native)

# Create a 'SimulatorImaging' object (contains effects that occur when imaging data is acquired in a telescope)
simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,  # Diffraction due to the telescope optics
    background_sky_level=0.1,  # Background light observed in addition to the strong lens's light
    # resulting image has this subtracted => simply acts as a source of noise
    add_poisson_noise=True  # Due to the background sky, lens galaxy and source galaxy Poisson photon counts
)
# Apply the simulator to the image
imaging = simulator.via_tracer_from(tracer=tracer, grid=grid)

# Plot the resulting simulated image (blurred due to the telescope optics & noise has been added)
imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.figures_2d(image=True)

# --- Output the simulated data (to '.fits' files) ---
dataset_name = "Clara_Model1"
dataset_path = path.join("dataset", "SIMULATIONS", dataset_name)  # where the data is output
imaging.output_to_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    overwrite=True,
)
