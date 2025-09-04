## Datasets

This folder contains all datasets used in the final project.

### Original Dataset

The dataset that will be explored in this project is the "Synthetic Indian Automobile Crash Data"^[This dataset was sourced from Swayam Patil's "Synthetic Indian Automobile Crash Data" dataset on [Kaggle.com](https://www.kaggle.com/datasets/swish9/synthetic-indian-automobile-crash-data).] dataset. It is a simulated dataset containing variables relevant to automobile crashes and safety features in India. Some key factors that were included were the vehicle characteristics (manufacturer, type, year of manufacture, weight, etc.), safety rating, number of airbags, crash statistics, and driver information.


- `automobile_crash.csv`: original dataset downloaded from source
- `automobile_crash_codebook.rda`: codebook of the cleaned dataset


### Cleaned Dataset

- `automobile_crash_clean.rda`: cleaned dataset ready for data set-ups

### Data Set-Up

- `crash_split.rda`: splitted data
- `crash_train.rda`: training dataset
- `crash_test.rda`: testing dataset
- `crash_folds.rda`: v-folds of the training dataset
