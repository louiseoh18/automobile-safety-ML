# model analysis and selection

# Load package(s)
library(tidymodels)
library(tidyverse)
library(here)
library(knitr)

# handle common conflicts
tidymodels_prefer()

set.seed(100)

# load data
load(here("data/crash_train.rda"))
load(here("data/crash_folds.rda"))


# basic models
load(here("results/fit_log_basic.rda"))
load(here("results/fit_log_complex.rda"))
load(here("results/tune_en_basic.rda"))
load(here("results/tune_en_complex.rda"))
load(here("results/tune_knn_basic.rda"))
load(here("results/tune_knn_complex.rda"))
load(here("results/tune_bt_basic.rda"))
load(here("results/tune_bt_complex.rda"))
load(here("results/tune_rf_basic.rda"))
load(here("results/tune_rf_complex.rda"))
# new models
load(here("results/tune_mars_basic.rda"))
load(here("results/tune_mars_complex.rda"))
load(here("results/tune_nn_basic.rda"))
load(here("results/tune_nn_complex.rda"))
load(here("results/tune_svm_poly_basic.rda"))
load(here("results/tune_svm_poly_complex.rda"))
load(here("results/tune_svm_radial_basic.rda"))
load(here("results/tune_svm_radial_complex.rda"))

# Load trained ensemble model info
load(here("results/crash_ensemble_model.rda"))
load(here("results/crash_pred.rda"))
load(here("results/member_pred.rda"))

###############################################################################
# Assess trained ensemble model

ensemble_roc_tbl <- roc_auc(crash_pred, truth = crash_severity, .pred_not_severe) |> 
  mutate(model = "ensemble",
         recipe = "NA",
         std_err = "NA",
         runtime = tictoc_ensemble$runtime
  ) |> 
  rename(ROC_AUC = .estimate) |> 
  select(model, recipe, ROC_AUC, std_err, runtime) |> 
  mutate(std_err = as.numeric(std_err))
         
roc_auc(crash_pred, truth = crash_severity, .pred_not_severe)

##############################################################################

# ROC AUC
model_results <- as_workflow_set(
  `log basic` = fit_log_basic,
  `log complex` = fit_log_complex,
  `en basic` = tune_en_basic,
  `en complex` = tune_en_complex,
  `knn basic` = tune_knn_basic,
  `knn complex` = tune_knn_complex,
  `bt basic` = tune_bt_basic,
  `bt complex` = tune_bt_complex,
  `rf basic` = tune_rf_basic,
  `rf complex` = tune_rf_complex,
  `mars basic` = tune_mars_basic,
  `mars complex` = tune_mars_complex,
  `nn basic` = tune_nn_basic,
  `nn complex` = tune_nn_complex,
  `svm poly basic` = tune_svm_poly_basic,
  `svm poly complex` = tune_svm_poly_complex,
  `svm radial basic` = tune_svm_radial_basic,
  `svm radial complex` = tune_svm_radial_complex
)

tbl_best_roc <- model_results |>
  collect_metrics() |>
  filter(.metric == "roc_auc") |>
  slice_max(mean, by = wflow_id) |> 
  # arrange(desc(mean)) |> 
  distinct(wflow_id, .keep_all = TRUE) |> 
  mutate(model = c("logistic", "logistic", 
                   "elastic net", "elastic net", 
                   "k nearest neighbor", "k nearest neighbor", 
                   "boosted tree", "boosted tree", 
                   "random forest", "random forest",
                   "mars", "mars", "nn", "nn",
                   "svm poly", "svm poly", "svm radial", "svm radial"),
  recipe = c("basic", "complex", "basic", "complex", 
             "basic", "complex", "basic", "complex", 
             "basic", "complex", "basic", "complex",
             "basic", "complex", "basic", "complex",
             "basic", "complex")) |>
  select(model, recipe, mean, std_err, n) 

tbl_runtime <- bind_rows(tictoc_log_basic, tictoc_log_complex,
                         tictoc_en_basic, tictoc_en_complex,
                         tictoc_knn_basic, tictoc_knn_complex,
                         tictoc_bt_basic, tictoc_bt_complex,
                         tictoc_rf_basic, tictoc_rf_complex,
                         tictoc_mars_basic, tictoc_mars_complex,
                         tictoc_nn_basic, tictoc_nn_complex,
                         tictoc_svm_poly_basic, tictoc_svm_poly_complex,
                         tictoc_svm_radial_basic, tictoc_svm_radial_complex) |> 
  mutate(model = c("logistic", "logistic", 
                   "elastic net", "elastic net", 
                   "k nearest neighbor", "k nearest neighbor", 
                   "boosted tree", "boosted tree", 
                   "random forest", "random forest",
                   "mars", "mars", "nn", "nn",
                   "svm poly", "svm poly", "svm radial", "svm radial"),
  recipe = c("basic", "complex", "basic", "complex", 
             "basic", "complex", "basic", "complex", 
             "basic", "complex", "basic", "complex",
             "basic", "complex", "basic", "complex",
             "basic", "complex")) |>
  select(model, recipe, runtime)

tbl_analysis <- merge(tbl_best_roc, tbl_runtime,
                      by = c("model", "recipe")) |> 
  # arrange(mean) |> 
  select(-n) |> 
  rename(ROC_AUC = mean) 
  # knitr::kable(caption = "ROC AUC and Runtime of Models")

tbl_analysis <- bind_rows(tbl_analysis, ensemble_roc_tbl) |> 
  arrange(desc(ROC_AUC))

# save table
save(tbl_analysis, file = here("analysis/model_roc_auc.rda"))

########################################

## AUTOPLOT -----

# en
en_basic_auto <- autoplot(tune_en_basic, metric = "roc_auc") + 
  theme_minimal() +
  labs(title = "Elastic Net",
       subtitle = "basic recipe")
en_complex_auto <- autoplot(tune_en_complex, metric = "roc_auc") +
theme_minimal() +
  labs(title = "Elastic Net",
       subtitle = "complex recipe")

# knn
knn_basic_auto <- autoplot(tune_knn_basic, metric = "roc_auc") + 
  theme_minimal() +
  labs(title = "K Nearest Neighbor",
       subtitle = "basic recipe")
knn_complex_auto <- autoplot(tune_knn_complex, metric = "roc_auc") +
  theme_minimal() +
  labs(title = "K Nearest Neighbor",
       subtitle = "complex recipe")

# bt
bt_basic_auto <- autoplot(tune_bt_basic, metric = "roc_auc") + 
  theme_minimal() +
  labs(title = "Boosted Tree",
       subtitle = "basic recipe")
bt_complex_auto <- autoplot(tune_bt_complex, metric = "roc_auc") +
  theme_minimal() +
  labs(title = "Boosted Tree",
       subtitle = "complex recipe")

# rf
rf_basic_auto <- autoplot(tune_rf_basic, metric = "roc_auc") + 
  theme_minimal() +
  labs(title = "Random Forest",
       subtitle = "basic recipe")
rf_complex_auto <- autoplot(tune_rf_complex, metric = "roc_auc") +
  theme_minimal() +
  labs(title = "Random Forest",
       subtitle = "complex recipe")

# nn
nn_basic_auto <- autoplot(tune_nn_basic, metric = "roc_auc") + 
  theme_minimal() +
  labs(title = "Neural Network",
       subtitle = "basic recipe")

# mars
mars_basic_auto <- autoplot(tune_mars_basic, metric = "roc_auc") + 
  theme_minimal() +
  labs(title = "MARS",
       subtitle = "basic recipe")

# svm rbf
rbf_basic_auto <- autoplot(tune_svm_radial_basic, metric = "roc_auc") + 
  theme_minimal() +
  labs(title = "SVM RBF",
       subtitle = "basic recipe")

# svm poly
poly_basic_auto <- autoplot(tune_svm_poly_basic, metric = "roc_auc") + 
  theme_minimal() +
  labs(title = "SVM Polynomial",
       subtitle = "basic recipe")

ensemble_auto <- autoplot(roc_curve(crash_pred, truth = crash_severity, .pred_not_severe))

# save model tuning para analysis autoplots

save(en_basic_auto, en_complex_auto,
     file = here("analysis/en_model_autoplot.rda"))

save(knn_basic_auto, knn_complex_auto,
     file = here("analysis/knn_model_autoplot.rda"))

save(bt_basic_auto,
     file = here("analysis/bt_basic_model_autoplot.rda"))

save(bt_complex_auto, 
     file = here("analysis/bt_complex_model_autoplot.rda"))

save(rf_basic_auto, rf_complex_auto,
     file = here("analysis/rf_model_autoplot.rda"))

save(nn_basic_auto, mars_basic_auto,
     rbf_basic_auto, poly_basic_auto,
     file = here("analysis/appendix_autoplot.rda"))

save(ensemble_auto,
     file = here("analysis/ensemble_autoplot.rda"))

## TUNED MODEL PARAMETERS -----

# en
en_basic_best <- select_best(tune_en_basic, metric = "roc_auc") |> 
  mutate(recipe = "basic recipe") 
en_complex_best <- select_best(tune_en_complex, metric = "roc_auc") |>
  mutate(recipe = "complex recipe")
en_best <- bind_rows(en_basic_best, en_complex_best) |>
  select(recipe, everything(), -.config)
  # kable(caption = "Elastic Net")

# knn
knn_basic_best <- select_best(tune_knn_basic, metric = "roc_auc") |> 
  mutate(recipe = "basic recipe") 
knn_complex_best <- select_best(tune_knn_complex, metric = "roc_auc") |>
  mutate(recipe = "complex recipe")
knn_best <- bind_rows(knn_basic_best, knn_complex_best) |>
  select(recipe, everything(), -.config)
  # kable(caption = "Elastic Net")

# bt
bt_basic_best <- select_best(tune_bt_basic, metric = "roc_auc") |> 
  mutate(recipe = "basic recipe") 
bt_complex_best <- select_best(tune_bt_complex, metric = "roc_auc") |>
  mutate(recipe = "complex recipe")
bt_best <- bind_rows(bt_basic_best, bt_complex_best) |>
  select(recipe, everything(), -.config)
  # kable(caption = "Boosted Tree")

# rf
rf_basic_best <- select_best(tune_rf_basic, metric = "roc_auc") |> 
  mutate(recipe = "basic recipe") 
rf_complex_best <- select_best(tune_rf_complex, metric = "roc_auc") |>
  mutate(recipe = "complex recipe")
rf_best <- bind_rows(rf_basic_best, rf_complex_best) |>
  select(recipe, everything(), -.config)
  # kable(caption = "Random Forest")

# nn
nn_basic_best <- select_best(tune_nn_basic, metric = "roc_auc") |> 
  mutate(recipe = "basic recipe") 
nn_complex_best <- select_best(tune_nn_complex, metric = "roc_auc") |>
  mutate(recipe = "complex recipe")
nn_best <- bind_rows(nn_basic_best, nn_complex_best) |>
  select(recipe, everything(), -.config)

# mars
mars_basic_best <- select_best(tune_mars_basic, metric = "roc_auc") |> 
  mutate(recipe = "basic recipe") 
mars_complex_best <- select_best(tune_mars_complex, metric = "roc_auc") |>
  mutate(recipe = "complex recipe")
mars_best <- bind_rows(mars_basic_best, mars_complex_best) |>
  select(recipe, everything(), -.config)

# svm poly
poly_basic_best <- select_best(tune_svm_poly_basic, metric = "roc_auc") |> 
  mutate(recipe = "basic recipe") 
poly_complex_best <- select_best(tune_svm_poly_complex, metric = "roc_auc") |>
  mutate(recipe = "complex recipe")
poly_best <- bind_rows(poly_basic_best, poly_complex_best) |>
  select(recipe, everything(), -.config)

# svm rbf
rbf_basic_best <- select_best(tune_svm_radial_basic, metric = "roc_auc") |> 
  mutate(recipe = "basic recipe") 
rbf_complex_best <- select_best(tune_svm_radial_complex, metric = "roc_auc") |>
  mutate(recipe = "complex recipe")
rbf_best <- bind_rows(rbf_basic_best, rbf_complex_best) |>
  select(recipe, everything(), -.config)

# save model para tables
save(en_best, knn_best, bt_best, rf_best,
     nn_best, mars_best, poly_best, rbf_best,
     file = here("analysis/model_tuning.rda"))

########################################
