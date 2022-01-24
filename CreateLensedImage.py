# Importing packages
import os.path
import autolens
import autolens.plot

# Change directory to current one
# Ref: https://linuxize.com/post/python-get-change-current-working-directory/
cwd = os.getcwd()
os.chdir('/Users/Geraldine/PycharmProjects/PyAuto/autolens_workspace')
print("Current working directory: {0}".format(os.getcwd()))

# Grid
image_plane_grid = autolens.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

#-----------------------------------------------------------------------------------------------------------------------
# Galaxies (lens + source)
mass_profile = autolens.mp.EllIsothermal(centre=(0.0,0.0), elliptical_comps=(0.5,0.5),
                                         einstein_radius=2.5)
lens_galaxy = autolens.Galaxy(redshift=0.5, mass=mass_profile)
light_profile = autolens.lp.SphSersic(
    centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, sersic_index=1.0)
source_galaxy = autolens.Galaxy(redshift=1.0, bulge=light_profile)
#-----------------------------------------------------------------------------------------------------------------------

# Planes
image_plane = autolens.Plane(galaxies=[lens_galaxy])

# Deflections
deflections = image_plane.deflections_2d_from(grid=image_plane_grid)
plane_plotter = autolens.plot.PlanePlotter(plane=image_plane, grid=image_plane_grid)

# Ray tracing
source_plane_grid = image_plane.traced_grid_from(grid=image_plane_grid)
source_plane = autolens.Plane(galaxies=[source_galaxy])

# Tracer
tracer = autolens.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
tracer_plotter = autolens.plot.TracerPlotter(tracer=tracer, grid=image_plane_grid)
tracer_plotter.figures_2d(image=True)

# Lensed image
#plane_plotter = autolens.plot.PlanePlotter(plane=source_plane, grid=source_plane_grid)
#plane_plotter.figures_2d(image=True)