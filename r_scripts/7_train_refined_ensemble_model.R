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
load(here("results/tune_refined_bt_basic.rda"))
load(here("results/tune_refined_rf_basic.rda"))

# load training dataset
load(here("data/crash_train.rda"))

# parallel processing ----
library(doMC)
registerDoMC(cores = parallel::detectCores(logical = TRUE))

########################################

# Create data stack ----
# all models have to be run on the exact same folds or else stacks will fail!
# stack could pick multiple model components to put into the ensemble

refined_crash_stack <- stacks() |> 
  add_candidates(tune_refined_bt_basic) |>  
  add_candidates(tune_refined_rf_basic) 

# Fit the stack ----
# penalty values for blending (set penalty argument when blending)
# a penalty decides the amount of regularization
# a high penalty = more regularization = less candidate models (force more coefficients to 0)
# we're estimating the coefficients we want for each prediction
# remember: penalties can't be negative (manually defining a range > 0)
# spacing out the values of the penalties
blend_penalty <- c(10^(-3.5:-0.5))

# Blend predictions (tuning step, set seed)
set.seed(100)

# perform lasso selection on candidate models
refined_crash_stack_blend <- refined_crash_stack |> 
  blend_predictions(penalty = blend_penalty)

# show member and optimal parameters
refined_member_coef <- refined_crash_stack_blend |> 
  collect_parameters("tune_refined_bt_basic") |> 
  filter(coef != 0) |>
  select(member, coef) |> 
  rbind(refined_crash_stack_blend |> 
          collect_parameters("tune_refined_rf_basic") |> 
          filter(coef != 0) |> 
          select(member, coef)) |> 
  rename(`Member` = member,
         `Coef` = coef)

# Save blended model stack for reproducibility & easy reference (for report)
save(refined_crash_stack_blend, file = here("results/refined_crash_stack_blend.rda"))
save(refined_member_coef, file = here("results/refined_member_coef.rda"))

# Explore the blended model stack
autoplot(refined_crash_stack_blend) +
  theme_minimal()

# show how many members from each model
autoplot(refined_crash_stack_blend, type = "members") +
  theme_minimal()

# show optimal penalty with member coefficients
autoplot(refined_crash_stack_blend, type = "weights") +
  theme_minimal()

# fit to training set ----

tic.clearlog() # clear log
tic("Refined Ensemble: REC 1") # start clock

refined_crash_model <- refined_crash_stack_blend |> 
  fit_members()

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_refined_ensemble <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# Save trained ensemble model for reproducibility & easy reference (for report)
save(tictoc_refined_ensemble, refined_crash_model, file = here("results/refined_crash_ensemble_model.rda"))

refined_crash_pred <- crash_train |> 
  select(crash_severity) |> 
  bind_cols(predict(refined_crash_model, crash_train, type = "prob"))

# Save ensemble prediction
save(refined_crash_pred, file = here("results/refined_crash_pred.rda"))

# member predictions
refined_member_pred <- crash_train |> 
  select(crash_severity) |> 
  bind_cols(predict(refined_crash_model, crash_train, members = TRUE, type = "prob"))

# Save member predictions
save(refined_member_pred, file = here("results/refined_member_pred.rda"))

