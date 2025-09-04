# assess final model

# Load package(s)
library(tidyverse)
library(tidymodels)
library(here)
library(stacks)

# handle common conflicts
tidymodels_prefer()

set.seed(100)

# Load testing data
load(here("data/crash_test.rda"))

# Load trained ensemble model info
load(here("results/refined_crash_ensemble_model.rda"))

# parallel processing ----
library(doMC)
registerDoMC(cores = parallel::detectCores(logical = TRUE))


###############################################################################

# Assessing trained + refined ensemble model to testing

refined_crash_pred_test <- crash_test |> 
  select(crash_severity) |> 
  bind_cols(predict(refined_crash_model, crash_test, type = "prob"))

# Save ensemble prediction on testing
save(refined_crash_pred_test, file = here("results/refined_crash_pred_test.rda"))

# Assess final ensemble model
refined_ensemble_test_roc <- roc_auc(refined_crash_pred_test, truth = crash_severity, .pred_not_severe)

# ROC AUC Curve on Final Model
refined_ensemble_roc_curve <- autoplot(roc_curve(refined_crash_pred_test, crash_severity, .pred_not_severe))

refined_ensemble_auto <- autoplot(refined_crash_model)

save(refined_ensemble_roc_curve,
     refined_ensemble_test_roc,
     refined_ensemble_auto,
     file = here("analysis/refined_ensemble_autoplot.rda"))

# member predictions
refined_member_pred_test <- crash_test |> 
  select(crash_severity) |> 
  bind_cols(predict(refined_crash_model, crash_test, members = TRUE, type = "prob"))

# Save member predictions
save(refined_member_pred_test, file = here("results/refined_member_pred_test.rda"))

# Comparing to member models
roc_auc(refined_crash_pred_test, truth = crash_severity, .pred_not_severe) |> 
  rbind(roc_auc(refined_member_pred_test, truth = crash_severity, .pred_not_severe))


