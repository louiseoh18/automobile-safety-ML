# mars - complex

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
load(here("recipes/recipe_mars_fe.rda"))

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
  add_recipe(recipe_mars_fe)

# hyperparameter tuning
# check range and update params
mars_param <- hardhat::extract_parameter_set_dials(mars_model) |> 
  update(num_terms = num_terms(range = c(1, 5)))

recipe_param <- hardhat::extract_parameter_set_dials(recipe_mars_fe) |> 
  update(num_comp = num_comp(range = c(1, 10)))

# put recipe and model params into one object
all_param <- bind_rows(mars_param, recipe_param)

mars_grid <- grid_regular(all_param, levels = c(5, 2, 10))

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("MARS: REC 2") # start clock

# fit model
tune_mars_complex <- mars_wf |> 
  tune_grid(crash_folds, 
            grid = mars_grid,
            control = control_grid(save_workflow = TRUE, save_pred = TRUE),
            metrics = metric_set(roc_auc)            
            )

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_mars_complex <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

########################################

# save fit
save(tune_mars_complex, tictoc_mars_complex, file = here("results/tune_mars_complex.rda"))

show_notes(tune_mars_complex)

