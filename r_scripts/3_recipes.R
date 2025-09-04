# recipes

# load packages
library(tidyverse)
library(tidymodels)
library(here)

# set seed for random process
set.seed(100)

# handle common conflicts
tidymodels_prefer()

# load data
load(here("data/crash_train.rda"))
load(here("data/crash_folds.rda"))

########################################

# BASIC RECIPE PARAMETRIC

recipe_basic <- recipe(crash_severity ~., data = crash_train) |>
  # we have no unique ID
  # step_rm() |>
  step_impute_median(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_novel(all_nominal_predictors()) |>
  step_unknown(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

recipe_basic_prep <- recipe_basic |>
  prep() |>
  bake(new_data = NULL)

save(recipe_basic, file = here("recipes/recipe_basic.rda"))

# BASIC RECIPE NONPARAMETRIC

recipe_basic_tree <- recipe(crash_severity ~., data = crash_train) |>
  # we have no unique ID
  # step_rm() |>
  step_impute_median(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_novel(all_nominal_predictors()) |>
  step_unknown(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

# 55 predictor vars
recipe_basic_tree_prep <- recipe_basic_tree |>
  prep() |>
  bake(new_data = NULL)

save(recipe_basic_tree, file = here("recipes/recipe_basic_tree.rda"))

###############################################################################

# COMPLEX RECIPE PARAMETRIC

recipe_complex <- recipe(crash_severity ~., data = crash_train) |>
  # no variables with over 20% missingness
  step_impute_knn(day_of_week, weather_conditions, crash_location) |>
  step_impute_knn(road_surface_conditions, impute_with = imp_vars(weather_conditions)) |>
  step_impute_knn(abs_presence, esc_presence, tcs_presence, tpms_presence, 
                  impute_with = imp_vars(safety_rating)) |> 
  step_impute_mean(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_novel(all_nominal_predictors()) |>
  step_unknown(all_nominal_predictors()) |>
  step_other(all_nominal_predictors(), threshold = 0.05, other = "other") |>
  step_dummy(all_nominal_predictors()) |>
  step_nzv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

recipe_complex_prep <- recipe_complex |>
  prep() |>
  bake(new_data = NULL)

save(recipe_complex, file = here("recipes/recipe_complex.rda"))

# COMPLEX RECIPE PARAMETRIC

recipe_complex_tree <- recipe(crash_severity ~., data = crash_train) |>
  # no variables with over 20% missingness
  step_impute_knn(day_of_week, weather_conditions, crash_location) |>
  step_impute_knn(road_surface_conditions, impute_with = imp_vars(weather_conditions)) |>
  step_impute_knn(abs_presence, esc_presence, tcs_presence, tpms_presence, 
                  impute_with = imp_vars(safety_rating)) |> 
  step_impute_mean(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_novel(all_nominal_predictors()) |>
  step_unknown(all_nominal_predictors()) |>
  step_other(all_nominal_predictors(), threshold = 0.05, other = "other") |>
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
  step_nzv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

recipe_complex_tree_prep <- recipe_complex_tree |>
  prep() |>
  bake(new_data = NULL)

save(recipe_complex_tree, file = here("recipes/recipe_complex_tree.rda"))

###############################################################################

# COMPLEX RECIPE FOR MARS

recipe_mars_fe <- recipe(crash_severity ~ ., data = crash_train) |> 
  step_impute_knn(day_of_week, weather_conditions, crash_location) |>
  step_impute_knn(road_surface_conditions, impute_with = imp_vars(weather_conditions)) |>
  step_impute_knn(abs_presence, esc_presence, tcs_presence, tpms_presence, 
                  impute_with = imp_vars(safety_rating)) |> 
  step_impute_mean(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |> 
  step_interact(terms = ~all_predictors()*all_predictors()) |>
  step_zv(all_predictors()) |> 
  step_corr(all_predictors(), threshold = 0.9) |> 
  step_normalize(all_numeric_predictors()) |> 
  step_pca(all_predictors(), num_comp = tune()) 

tunable(recipe_mars_fe)

save(recipe_mars_fe, file = here("recipes/recipe_mars_fe.rda"))

###############################################################################

# BASIC RECIPE FOR NN

recipe_nn_basic <- recipe(crash_severity ~., data = crash_train) |>
  step_impute_mode(day_of_week, road_surface_conditions, weather_conditions,
                   crash_location, abs_presence, esc_presence, tcs_presence,
                   tpms_presence, driver_gender) |>
  step_other(all_nominal_predictors(), threshold = 0.05, other = "other") |> 
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
  step_nzv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

recipe_nn_basic_prep <- recipe_nn_basic |>
  prep() |>
  bake(new_data = NULL)

save(recipe_nn_basic, file = here("recipes/recipe_nn_basic.rda"))

# COMPLEX RECIPE FOR NN

recipe_nn_complex <- recipe(crash_severity ~., data = crash_train) |>
  step_impute_knn(day_of_week, road_surface_conditions, weather_conditions,
                   crash_location, abs_presence, esc_presence, tcs_presence,
                   tpms_presence, driver_gender) |>
  step_impute_mean(all_numeric_predictors()) |> 
  step_impute_mode(all_nominal_predictors()) |> 
  step_other(all_nominal_predictors(), threshold = 0.05, other = "other") |> 
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
  step_nzv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

recipe_nn_complex_prep <- recipe_nn_complex |>
  prep() |>
  bake(new_data = NULL)

save(recipe_nn_complex, file = here("recipes/recipe_nn_complex.rda"))

