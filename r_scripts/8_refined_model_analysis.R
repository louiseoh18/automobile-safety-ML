# model analysis and selection

# Load package(s)
library(tidymodels)
library(tidyverse)
library(here)
library(knitr)
library(stacks)

# handle common conflicts
tidymodels_prefer()

set.seed(100)

# load data
load(here("data/crash_train.rda"))
load(here("data/crash_folds.rda"))

# load refined models
load(here("results/tune_refined_rf_basic.rda"))
load(here("results/tune_refined_bt_basic.rda"))

# Load refined ensemble model info
load(here("results/refined_crash_ensemble_model.rda"))
load(here("results/refined_crash_pred.rda"))

###############################################################################
# Assess refined trained ensemble model ----

refined_ensemble_roc_tbl <- roc_auc(refined_crash_pred, truth = crash_severity, .pred_not_severe) |> 
  mutate(model = "ensemble",
         recipe = "NA",
         std_err = "NA",
         runtime = tictoc_refined_ensemble$runtime
  ) |> 
  rename(ROC_AUC = .estimate) |> 
  select(model, recipe, ROC_AUC, std_err, runtime) |> 
  mutate(std_err = as.numeric(std_err))

##############################################################################

# refined model analysis ----
model_results <- as_workflow_set(
  `refined bt basic` = tune_refined_bt_basic,
  `refined rf basic` = tune_refined_rf_basic
)

tbl_best_roc <- model_results |>
  collect_metrics() |>
  filter(.metric == "roc_auc") |>
  slice_max(mean, by = wflow_id) |> 
  distinct(wflow_id, .keep_all = TRUE) |> 
  mutate(model = c("Boosted Tree", "Random Forest"),
         recipe = "Kitchen Sink") |>
  select(model, recipe, mean, std_err, n) 

tbl_runtime <- bind_rows(refined_tictoc_bt_basic, refined_tictoc_rf_basic) |> 
  mutate(model = c("Boosted Tree", "Random Forest"),
         recipe = "Kitchen Sink") |>
  select(model, recipe, runtime)

refined_tbl_analysis <- merge(tbl_best_roc, tbl_runtime,
                      by = c("model", "recipe")) |> 
  select(-n) |> 
  rename(ROC_AUC = mean) 
# knitr::kable(caption = "ROC AUC and Runtime of Models")

refined_tbl_analysis <- bind_rows(refined_tbl_analysis, refined_ensemble_roc_tbl) |> 
  arrange(desc(ROC_AUC))

# save table
save(refined_tbl_analysis, file = here("analysis/refined_model_roc_auc.rda"))




