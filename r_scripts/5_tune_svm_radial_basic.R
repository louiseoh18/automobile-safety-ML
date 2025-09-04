# svm radial - basic

# Load package(s)
library(tidyverse)
library(tidymodels)
library(here)
library(tictoc)
library(kernlab)

# handle common conflicts
tidymodels_prefer()

set.seed(100)

# load data
load(here("data/crash_folds.rda"))
load(here("recipes/recipe_basic_tree.rda"))

# parallel processing ----
library(doMC)
registerDoMC(cores = parallel::detectCores(logical = TRUE))


########################################

# model specifications
svm_radial_model <- svm_rbf(
  mode = "classification", 
  cost = tune(),
  rbf_sigma = tune()
) |>
  set_engine("kernlab")

# define workflow
svm_radial_basic_wflow <- workflow() |> 
  add_model(svm_radial_model) |> 
  add_recipe(recipe_basic_tree)

# hyperparameter tuning
# check range and update params
svm_radial_basic_param <- hardhat::extract_parameter_set_dials(svm_radial_model)

# build tuning grid
svm_radial_basic_grid <- grid_latin_hypercube(svm_radial_basic_param, size = 50)

# tune/fit workflow/model ----

tic("SVM RADIAL BASIC: REC1") # start clock

tune_svm_radial_basic <- svm_radial_basic_wflow |> 
  tune_grid(
    resamples = crash_folds,
    grid = svm_radial_basic_grid,
    control = control_grid(save_workflow = TRUE, save_pred = TRUE),
    metrics = metric_set(roc_auc)
  )

toc(log = TRUE) # stop clock

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_svm_radial_basic <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

########################################

# save fit
save(tune_svm_radial_basic, tictoc_svm_radial_basic,
     file = here("results/tune_svm_radial_basic.rda"))

show_notes(tune_svm_radial_basic)

