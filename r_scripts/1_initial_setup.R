# INITIAL SETUP - DATA SPLITTING AND FOLDS

# load library
library(tidyverse)
library(tidymodels)
library(here)

# set seed for random process
set.seed(100)

# handle common conflicts
tidymodels_prefer()

# load clean data
load(here("data/automobile_crash_clean.rda"))

#################################################################
# exploring distribution of target variable: crash_severity

crash_data |> 
  ggplot(aes(x = crash_severity)) +
  geom_bar(fill = "skyblue") +
  labs(
    y = NULL,
    x = "Crash Severity",
    title = "Distribution of Car Crash Severity Outcomes"
  ) +
  theme_minimal()

crash_data |> 
  count(crash_severity)

#################################################################

# split data
crash_split <- initial_split(crash_data, prop = 0.8, strata = crash_severity)
crash_train <- training(crash_split)
crash_test <- testing(crash_split)

# set up folds
crash_folds <- vfold_cv(crash_train, v = 5, repeats = 3, strata = crash_severity)

#################################################################

# save datasets
save(crash_split, crash_train, crash_test, file = here("data/crash_split.rda"))
save(crash_train, file = here("data/crash_train.rda"))
save(crash_test, file = here("data/crash_test.rda"))
save(crash_folds, file = here("data/crash_folds.rda"))

