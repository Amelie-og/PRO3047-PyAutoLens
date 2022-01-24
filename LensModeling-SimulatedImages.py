"""
This script is used to test the program with its own images by doing lens modeling (non-linear search) using images we
created/simulated with the software itself and providing the correct/wrong model to see what happens.
(Similar to LensModeling.py script, but uses simulated data instead of real data, and provides output on the validity
of the model in order to assess if the software is good at recognizing its own shortcomings).
References: See notebooks-imaging-modeling -> results.ipynb
"""

# Import packages
from os import path
import sys
import autolens as al
import autolens.plot as aplt
import autofit as af

# Load the (simulated) dataset
dataset_name = "Clara_Model1"
dataset_path = path.join("dataset", "SIMULATIONS", dataset_name)
imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,  # use the same as the one used to simulate the data
)

# Plot dataset (Image, Noise-Map, Point Spread Function, Signal-To-Noise Map, Inversion Noise-Map,
# Potential Chi-Squared Map)
imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

# --- Non-linear Search to do the lens modeling ---

# Create and apply the appropriate mask
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    radius=6.0  # TODO: Adapt this to the image!
)
imaging = imaging.apply_mask(mask=mask)
imaging_plotter = aplt.ImagingPlotter(
    imaging=imaging,
    visuals_2d=aplt.Visuals2D(mask=mask))
imaging_plotter.subplot_imaging()

# Create the final model (with source and lens models)
# TODO: Can provide correct/wrong model to see what happens (we know the model used for the simulation)
model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=al.mp.EllDevVaucouleurs),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.SphSersic), )
)

# Create the non-linear search object which will fit the lens model
output_path = "TESTING_Simulations"
name_of_search = "Clara_Model1_correct-model-70"
search = af.DynestyStatic(
    path_prefix=path.join(output_path),  # Where the search outputs the results (autolens_workspace/output/...)
    name=name_of_search,
    unique_tag=dataset_name,
    nlive=70,
    number_of_cores=1,
)

# Create the analysis object which defines how the search fits each model to the Imaging dataset
analysis = al.AnalysisImaging(dataset=imaging)

# Perform the non-linear search to find which model fits the data with the highest likelihood
print("Dynesty has begun running - checkout the autolens_workspace/output/ folder for live output of the results, "
      "images and lens model.")
result = search.fit(model=model, analysis=analysis)
print("Dynesty has successfully finished running, YAY!")

# --- Output the results (to assess quality/accuracy of the model)  ---

# Plot the result of the search to inspect how good the fit was
fit_imaging_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_imaging_plotter.subplot_fit_imaging()

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
