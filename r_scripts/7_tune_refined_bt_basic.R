# boosted tree - basic

# Load package(s)
library(tidymodels)
library(tidyverse)
library(here)
library(tictoc)
library(xgboost)

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
# load(here("analysis/bt_basic_model_autoplot.rda"))
# load(here("analysis/model_tuning.rda"))

# for bt_basic, optimal params 
# mtry = 20, min_n = 2, learn_rate = 0.00422
# bt_best

# increase max range for mtry => maybe c(18, 38)
# restrict range of min_n => c(1, 4)
# learn rate: update to c(-3, -1.5)
# bt_basic_auto

########################################

# model specification
bt_model <- boost_tree(trees = 1100,
                       min_n = tune(),
                       mtry = tune(),
                       learn_rate = tune()) |> 
  set_engine("xgboost") |> 
  set_mode("classification")

# define workflow
bt_wf <- workflow() |> 
  add_model(bt_model) |> 
  add_recipe(recipe_basic_tree)

# hyperparameter tuning
# change hyperparameter ranges
bt_param <- hardhat::extract_parameter_set_dials(bt_model) |> 
  update(min_n = min_n(c(1, 4)),
         mtry = mtry(c(18, 38)),
         learn_rate = learn_rate(c(-3, -1.5)))

# build tuning grid
bt_grid <- grid_regular(bt_param, levels = 10)

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("Refined BT: REC 1") # start clock

# fit model
tune_refined_bt_basic <- bt_wf |> 
  tune_grid(crash_folds, 
            grid = bt_grid,
            control = control_grid(save_workflow = TRUE, save_pred = TRUE),
            metrics = metric_set(roc_auc)
  )

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

refined_tictoc_bt_basic <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

########################################

# save fit
save(tune_refined_bt_basic, refined_tictoc_bt_basic, file = here("results/tune_refined_bt_basic.rda"))

show_notes(tune_refined_bt_basic)
