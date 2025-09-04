# k nearest neighbor - basic

# Load package(s)
library(tidymodels)
library(tidyverse)
library(here)
library(tictoc)
library(kknn)

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
knn_model <- nearest_neighbor(mode = "classification", 
                              neighbors = tune()) |> 
  set_engine("kknn")

# define workflow
knn_wf <- workflow() |> 
  add_model(knn_model) |> 
  add_recipe(recipe_basic_tree)

# hyperparameter tuning
# change hyperparameter ranges
knn_param <- hardhat::extract_parameter_set_dials(knn_model) |> 
  update(neighbors = neighbors(c(1, 20)))
# need to update neighbors

# build tuning grid
knn_grid <- grid_regular(knn_param, levels = 5)

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("KNN: REC 1") # start clock

# fit model
tune_knn_basic <- knn_wf |> 
  tune_grid(crash_folds, 
            grid = knn_grid,
            control = control_grid(save_workflow = TRUE, save_pred = TRUE),
            metrics = metric_set(roc_auc)            
            )

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_knn_basic <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

########################################

# save fit
save(tune_knn_basic, tictoc_knn_basic, file = here("results/tune_knn_basic.rda"))

show_notes(tune_knn_basic)

