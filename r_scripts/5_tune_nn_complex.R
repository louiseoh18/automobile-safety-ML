# nn - complex

# Load package(s)
library(tidyverse)
library(tidymodels)
library(tictoc)
library(here)
library(nnet)

# handle common conflicts
tidymodels_prefer()

set.seed(100)

# load data
load(here("data/crash_folds.rda"))
load(here("recipes/recipe_nn_complex.rda"))

# parallel processing ----
library(doMC)
registerDoMC(cores = parallel::detectCores(logical = TRUE))


########################################

# model specifications
nn_model <- mlp(mode = "classification",
                hidden_units = tune(),
                penalty = tune()) |> 
  set_engine("nnet")

# define workflow
nn_wf <- workflow() |> 
  add_model(nn_model) |> 
  add_recipe(recipe_nn_complex)

# hyperparameter tuning
# check range and update params
nn_param <- hardhat::extract_parameter_set_dials(nn_model)

# build tuning grid
nn_grid <- grid_regular(nn_param, levels = 5)

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("NN: REC 2") # start clock

# fit model
tune_nn_complex <- nn_wf |> 
  tune_grid(crash_folds, 
            grid = nn_grid,
            control = control_grid(save_workflow = TRUE, save_pred = TRUE),
            metrics = metric_set(roc_auc))

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_nn_complex <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

########################################

# save fit
save(tune_nn_complex, tictoc_nn_complex, file = here("results/tune_nn_complex.rda"))

show_notes(tune_nn_complex)

