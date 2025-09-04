# logistic model

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
load(here("data/crash_train.rda"))
load(here("recipes/recipe_basic.rda"))
load(here("recipes/recipe_complex.rda"))

# parallel processing ----
library(doMC)
registerDoMC(cores = parallel::detectCores(logical = TRUE))

########################################

# USING BASIC RECIPE

# model specifications
log_model_basic <- logistic_reg() |> 
  set_mode("classification") |> 
  set_engine("glm")

# define workflow
log_wf_basic <- workflow() |> 
  add_model(log_model_basic) |> 
  add_recipe(recipe_basic)

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("Logistic: REC 1") # start clock

# fit model
fit_log_basic <- log_wf_basic |> 
  fit_resamples(
    resamples = crash_folds,
    control = control_resamples(save_workflow = TRUE, save_pred = TRUE),
    metrics = metric_set(roc_auc)
  )

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_log_basic <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# save fit
save(fit_log_basic, tictoc_log_basic, file = here("results/fit_log_basic.rda"))


###########################################################################

# USING COMPLEX RECIPE

# model specifications
log_model_complex <- logistic_reg() |>
  set_mode("classification") |>
  set_engine("glm")

# define workflow
log_wf_complex <- workflow() |>
  add_model(log_model_complex) |>
  add_recipe(recipe_complex)

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("Logistic: REC 2") # start clock

# fit model
fit_log_complex <- log_wf_complex |>
  fit_resamples(
    resamples = crash_folds,
    control = control_resamples(save_workflow = TRUE, save_pred = TRUE)
  )

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_log_complex <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# save fit
save(fit_log_complex, tictoc_log_complex, file = here("results/fit_log_complex.rda"))

