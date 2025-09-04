# EDA

# load library
library(tidyverse)
library(tidymodels)
library(here)
library(corrplot)
library(psych)

# set seed for random process
set.seed(100)

# handle common conflicts
tidymodels_prefer()

# load clean data
load(here("data/crash_train.rda"))

#################################################################

# target variable
ggplot(crash_train, aes(x = crash_severity)) +
  geom_bar(color = 'black', fill = "skyblue") +
  theme_minimal()

# univariate
vars <- crash_train |> 
  colnames()
for (var in vars) {
  if (is.numeric(crash_train[[var]])) {
    # histogram for numeric
    hist <- ggplot(crash_train, aes(x = !!as.name(var))) +
      geom_histogram() +
      theme_minimal()
    print(hist)
  } else {
    # barplot for non-numeric
    bar <- ggplot(crash_train, aes(x = !!as.name(var))) +
      geom_bar() +
      theme_minimal()
    print(bar)
  }
}

# bivariate
corr <- crash_train |> 
  select(where(is.numeric)) |> 
  cor()
ggcorrplot::ggcorrplot(corr, 
                       type = "lower",
                       lab = TRUE, 
                       method = "square")

# categorical
lowerCor(crash_train) |>
  corrplot()

# missingness
miss_table <- naniar::miss_var_summary(crash_train) |>
  filter(n_miss > 0)

miss_names <- miss_table |> 
  pull(variable)

crash_train |> 
  select(all_of(miss_names)) |> 
  gg_miss_var()

#################################################################



