"""
Performing a Non-Linear Search to do the model-fitting of the data and find which lens model corresponds to it.
Steps: Load the data, create the imaging object, apply the mask, create the model, create the search, create the
analysis, perform the non-linear search, plot and print the results.
"""

# Import packages
from os import path
import sys
import autolens as al
import autolens.plot as aplt
import autofit as af

# Load and resize the image of the dataset
dataset_name = "B1938-ER"
dataset_path = path.join("dataset", "CASTLES", dataset_name)
image = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "image.fits"),
    pixel_scales=0.1 # TODO: change this for different images! (specific to detector)
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

# Create and apply the appropriate mask
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    radius=4.0 # TODO: Adapt this to the image!
)
imaging = imaging.apply_mask(mask=mask)
imaging_plotter = aplt.ImagingPlotter(
    imaging=imaging,
    visuals_2d=aplt.Visuals2D(mask=mask))
imaging_plotter.subplot_imaging()

# Create the final model (with source and lens models)
# TODO: Can change this to make it more realistic
model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.881, mass=al.mp.SphIsothermal, bulge=al.lp.SphSersic),
        source=af.Model(al.Galaxy, redshift=2.059, bulge=al.lp.SphSersic),)
)

# Create the non-linear search object which will fit the lens model
output_path = "CASTLES"
name_of_search = "B1938_non_linear_search_1"
search = af.DynestyStatic(
    path_prefix=path.join(output_path),  # Where the search outputs the results (autolens_workspace/output/...)
    name=name_of_search,
    unique_tag=dataset_name,
    nlive=40,
    number_of_cores=1,
)

# Create the analysis object which defines how the search fits each model to the Imaging dataset
analysis = al.AnalysisImaging(dataset=imaging)

# Perform the non-linear search to find which model fits the data with the highest likelihood
print("Dynesty has begun running - checkout the autolens_workspace/output/ folder for live output of the results, "
      "images and lens model.")
result = search.fit(model=model, analysis=analysis)
print("Dynesty has successfully finished running, YAY!")

# Plot the result of the search to inspect how good the fit was
fit_imaging_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_imaging_plotter.subplot_fit_imaging()

# Print the log likelihood of the result
print("The log_likelihood for this model is: ", result.max_log_likelihood_fit.log_likelihood)

# Plot the Probability Density Functions (PDF) of the results
dynesty_plotter = aplt.DynestyPlotter(samples=result.samples)
dynesty_plotter.cornerplot()

# Create a text file to compile the info on the results
text_path = path.join("output", output_path, dataset_name, name_of_search, "Results.txt")
original_stdout = sys.stdout
with open(text_path, "a") as f:  # open file for writing or create if it does not exist
    sys.stdout = f
    # Print the log likelihood of the result
    print("=== MODEL STATISTICS === \r\n")
    print("Log_likelihood of the resulting model: ", result.max_log_likelihood_fit.log_likelihood, "\n"
          , "--> The log likelihood is evaluated from the likelihood function (e.g. -0.5 * chi_squared + the noise "
            "normalization).\n")
    # Print the log prior, log posterior, and weight
    samples = result.samples
    res = max(samples.log_likelihood_list)
    res_idx = samples.log_likelihood_list.index(res)
    print("Log_prior of the resulting model: ", samples.log_prior_list[res_idx], "\n",
          "--> The log prior encodes information on how the priors on the parameters maps the log likelihood value to "
          "the logposterior value.\n")
    print("Log_posterior of the resulting model: ", samples.log_posterior_list[res_idx], "\n",
          "--> The log posterior is log_likelihood + log_prior.\n")
    print("Weight of the resulting model: ", samples.weight_list[res_idx], "\n",)
    # Print the Bayesian log evidence of the model fit
    print("Bayesian log evidence of the model fit: ", samples.log_evidence, "\n",
          "--> Bayesian log evidence is estimated via the nested sampling algorithm.\r\n")
    # Print values of the parameters
    ml_instance = samples.max_log_likelihood_instance
    print("=== MODEL PARAMETERS === \r\n")
    print("Estimated values for the parameters of the LENS galaxy: \n", ml_instance.galaxies.lens, "\r\n")
    print("Estimated values for the parameters of the SOURCE galaxy: \n", ml_instance.galaxies.source, "\r\n")
    # Print errors of the parameters (not possible unfortunately...)
    # Print model parameters at a given sigma value
    print("=== MODEL PARAMETERS AT GIVEN SIGMA VALUE === \r\n")
    print(samples.model.model_component_and_parameter_names, "\r\n")
        # Print values at 1 sigma (upper and lower values at 1 sigma confidence)
    uv1_vector = samples.vector_at_upper_sigma(sigma=1.0)
    lv1_vector = samples.vector_at_lower_sigma(sigma=1.0)
    print("Model parameters at a 1 sigma confidence (68.27%): \n")
    print("Upper values: \n", uv1_vector, "\n")
    print("Lower values: \n", lv1_vector, "\r\n")
        # Print values at 2 sigma (upper and lower values at 2 sigma confidence)
    uv2_vector = samples.vector_at_upper_sigma(sigma=2.0)
    lv2_vector = samples.vector_at_lower_sigma(sigma=2.0)
    print("Model parameters at a 2 sigma confidence (95.45%): \n")
    print("Upper values: \n", uv2_vector, "\n")
    print("Lower values: \n", lv2_vector, "\r\n")
        # Print values at 3 sigma (upper and lower values at 3 sigma confidence)
    uv3_vector = samples.vector_at_upper_sigma(sigma=3.0)
    lv3_vector = samples.vector_at_lower_sigma(sigma=3.0)
    print("Model parameters at a 3 sigma confidence (99.73%): \n")
    print("Upper values: \n", uv3_vector, "\n")
    print("Lower values: \n", lv3_vector, "\r\n")
    sys.Stdout = original_stdout