# random forest - basic

# Load package(s)
library(tidymodels)
library(tidyverse)
library(here)
library(tictoc)
library(ranger)

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

# model specification
rf_model <- rand_forest(mode = "classification",
                        trees = tune(),
                        min_n = tune(),
                        mtry = tune()) |> 
  set_engine("ranger")

# define workflow
rf_wf <- workflow() |> 
  add_model(rf_model) |> 
  add_recipe(recipe_basic_tree)

# hyperparameter tuning
# change hyperparameter ranges
rf_param <- hardhat::extract_parameter_set_dials(rf_model) |> 
  update(min_n = min_n(c(2, 20)),
         mtry = mtry(c(2, 20)))
# need to update min_n, mtry

# build tuning grid
rf_grid <- grid_regular(rf_param, levels = 5)

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("RF: REC 1") # start clock

# fit model
tune_rf_basic <- rf_wf |> 
  tune_grid(crash_folds, 
            grid = rf_grid,
            control = control_grid(save_workflow = TRUE, save_pred = TRUE),
            metrics = metric_set(roc_auc)            
            )

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_rf_basic <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)


# save fit
save(tune_rf_basic, tictoc_rf_basic, file = here("results/tune_rf_basic.rda"))

show_notes(tune_rf_basic)

