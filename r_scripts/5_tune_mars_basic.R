# mars - basic

# Load package(s)
library(tidyverse)
library(tidymodels)
library(tictoc)
library(here)
library(earth)

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
mars_model <- mars(mode = "classification",
                 num_terms = tune(),
                 prod_degree = tune()) |> 
  set_engine("earth")

# define workflow
mars_wf <- workflow() |> 
  add_model(mars_model) |> 
  add_recipe(recipe_basic)

# hyperparameter tuning
# check range and update params
mars_param <- hardhat::extract_parameter_set_dials(mars_model) |> 
  update(num_terms = num_terms(range = c(1, 5)))

# build tuning grid
mars_grid <- grid_regular(mars_param, levels = 5)

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("MARS: REC 1") # start clock

# fit model
tune_mars_basic <- mars_wf |> 
  tune_grid(crash_folds, 
            grid = mars_grid,
            control = control_grid(save_workflow = TRUE, save_pred = TRUE),
            metrics = metric_set(roc_auc)            
            )

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_mars_basic <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

########################################

# save fit
save(tune_mars_basic, tictoc_mars_basic, file = here("results/tune_mars_basic.rda"))

show_notes(tune_mars_basic)

