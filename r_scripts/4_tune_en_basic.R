# elastic net - basic

# Load package(s)
library(tidymodels)
library(tidyverse)
library(here)
library(tictoc)
library(glmnet)

# handle common conflicts
tidymodels_prefer()

set.seed(100)

# load data
load(here("data/crash_folds.rda"))
load(here("recipes/recipe_basic.rda"))

# parallel processing ----
library(doMC)
registerDoMC(cores = parallel::detectCores(logical = TRUE))


########################################

# model specifications
en_model <- logistic_reg(mixture = tune(),
                         penalty = tune()) |> 
  set_engine("glmnet") |> 
  set_mode("classification")

# define workflow
en_wf <- workflow() |> 
  add_model(en_model) |> 
  add_recipe(recipe_basic)

# hyperparameter tuning
# check range and update params
en_param <- hardhat::extract_parameter_set_dials(en_model) |> 
  update(mixture = mixture(c(0, 1)))
# need to update penalty

# build tuning grid
en_grid <- grid_regular(en_param, levels = 5)

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("EN: REC 1") # start clock

# fit model
tune_en_basic <- en_wf |> 
  tune_grid(crash_folds, 
            grid = en_grid,
            control = control_grid(save_workflow = TRUE, save_pred = TRUE),
            metrics = metric_set(roc_auc)            
            )

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_en_basic <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

########################################

# save fit
save(tune_en_basic, tictoc_en_basic, file = here("results/tune_en_basic.rda"))

show_notes(tune_en_basic)

