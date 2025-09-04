# refined random forest - basic

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

# thinking about model refinement

# 55 max predictors in recipe_basic_tree
# 
# load(here("analysis/rf_model_autoplot.rda"))
# load(here("analysis/model_tuning.rda"))

# for bt_basic, optimal params 
# mtry = 20, min_n = 20, trees = 2000
# select_best(tune_rf_basic, metric = "roc_auc")
# increase max range for mtry => maybe (19, 40)
# increase range of min_n => maybe(19, 40)
# trees: increase range of trees => maybe c(1975, 2200)
# rf_basic_auto

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
  update(min_n = min_n(c(19, 40)),
         mtry = mtry(c(19, 40)),
         trees = trees(c(1975, 2200)))
# need to update min_n, mtry

# build tuning grid
rf_grid <- grid_regular(rf_param, levels = 8)

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("Refined RF: REC 1") # start clock

# fit model
tune_refined_rf_basic <- rf_wf |> 
  tune_grid(crash_folds, 
            grid = rf_grid,
            control = control_grid(save_workflow = TRUE, save_pred = TRUE),
            metrics = metric_set(roc_auc)            
  )

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

refined_tictoc_rf_basic <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)


# save fit
save(tune_refined_rf_basic, refined_tictoc_rf_basic, file = here("results/tune_refined_rf_basic.rda"))

show_notes(tune_refined_rf_basic)

