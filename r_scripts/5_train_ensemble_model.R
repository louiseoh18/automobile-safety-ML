# ensemble

# Load package(s)
library(tidyverse)
library(tidymodels)
library(tictoc)
library(here)
library(stacks)

# handle common conflicts
tidymodels_prefer()

set.seed(100)

# load candidate model info
load(here("results/tune_bt_basic.rda"))
load(here("results/tune_rf_basic.rda"))
load(here("results/tune_nn_basic.rda"))
load(here("results/tune_en_basic.rda"))

# load training dataset
load(here("data/crash_train.rda"))

# parallel processing ----
library(doMC)
registerDoMC(cores = parallel::detectCores(logical = TRUE))


########################################

# Create data stack ----
# all models have to be run on the exact same folds or else stacks will fail!
# stack could pick multiple model components to put into the ensemble

crash_stack <- stacks() |> 
  add_candidates(tune_bt_basic) |>  
  add_candidates(tune_rf_basic) |>  
  add_candidates(tune_nn_basic) |>
  add_candidates(tune_en_basic)

# Fit the stack ----
# penalty values for blending (set penalty argument when blending)
# a penalty decides the amount of regularization
# a high penalty = more regularization = less candidate models (force more coefficients to 0)
# we're estimating the coefficients we want for each prediction
# remember: penalties can't be negative (manually defining a range > 0)
# spacing out the values of the penalties
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)

# Blend predictions (tuning step, set seed)
set.seed(100)

# perform lasso selection on candidate models
crash_stack_blend <- crash_stack |> 
  blend_predictions(penalty = blend_penalty)

# show member and optimal parameters
member_coef <- crash_stack_blend |> 
  collect_parameters("tune_bt_basic") |> 
  filter(coef != 0) |>
  select(member, coef) |> 
  rbind(crash_stack_blend |> 
          collect_parameters("tune_rf_basic") |> 
          filter(coef != 0) |> 
          select(member, coef)) |> 
  rbind(crash_stack_blend |> 
          collect_parameters("tune_nn_basic") |> 
          filter(coef != 0) |> 
          select(member, coef)) |> 
  rbind(crash_stack_blend |> 
          collect_parameters("tune_en_basic") |> 
          filter(coef != 0) |> 
          select(member, coef)) |> 
  rename(`Member` = member,
         `Coef` = coef)

# Save blended model stack for reproducibility & easy reference (for report)
save(crash_stack_blend, file = here("results/crash_stack_blend.rda"))
save(member_coef, file = here("results/member_coef.rda"))

# Explore the blended model stack
autoplot(crash_stack_blend) +
  theme_minimal()

# show how many members from each model
autoplot(crash_stack_blend, type = "members") +
  theme_minimal()

# show optimal penalty with member coefficients
autoplot(crash_stack_blend, type = "weights") +
  theme_minimal()

# fit to training set ----

tic.clearlog() # clear log
tic("Ensemble: REC 1") # start clock

crash_model <- crash_stack_blend |> 
  fit_members()

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_ensemble <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# Save trained ensemble model for reproducibility & easy reference (for report)
save(tictoc_ensemble, crash_model, file = here("results/crash_ensemble_model.rda"))

crash_pred <- crash_train |> 
  select(crash_severity) |> 
  bind_cols(predict(crash_model, crash_train, type = "prob"))

# Save ensemble prediction
save(crash_pred, file = here("results/crash_pred.rda"))

# member predictions
member_pred <- crash_train |> 
  select(crash_severity) |> 
  bind_cols(predict(crash_model, crash_train, members = TRUE, type = "prob"))

# Save member predictions
save(member_pred, file = here("results/member_pred.rda"))

