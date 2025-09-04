
## R Scripts

This folder contains all R scripts that lead to the final project.

- `0_data_cleaning.R`: codes used to clean the original dataset ready for use
- `1_initial_setup.R`: target variable exploration and data folds
- `2_eda.R`: exploratory data analysis performed before creating the recipe and tuning models
- `3_recipes.R`: codes used to create the basic, complex, and customized recipes used for model tuning
- `4_fit_log.R`: codes used to fit the logistic regression model using both the basic and complex recipes
- `4_tune_en_basic.R`: codes used to tune the elastic net model using the basic recipe
- `4_tune_en_complex.R`: codes used to tune the elastic net model using the complex recipe
- `4_tune_knn_basic.R`: codes used to tune the k-nearest neighbors model using the basic recipe
- `4_tune_knn_complex.R`: codes used to tune the k-nearest neighbors model using the complex recipe

- `4_tune_bt_basic.R`: codes used to tune the boosted tree model using the basic recipe
- `4_tune_bt_complex.R`: codes used to tune the boosted tree model using the complex recipe

- `4_tune_rf_basic.R`: codes used to tune the random forest model using the basic recipe
- `4_tune_rf_complex.R`: codes used to tune the random forest model using the complex recipe

- `5_tune_mars_basic.R`: codes used to tune the MARS model using the basic recipe
- `5_tune_mars_complex.R`: codes used to tune the MARS model using the complex recipe
- `5_tune_nn_basic.R`: codes used to tune the neural net model using the basic recipe
- `5_tune_nn_complex.R`: codes used to tune the neural net model using the complex recipe

- `5_tune_svm_poly_basic.R`: codes used to tune the SVM polynomial model using the basic recipe
- `5_tune_svm_poly_complex.R`: codes used to tune the SVM polynomial model using the complex recipe
- `5_tune_svm_radial_basic.R`: codes used to tune the SVM radial model using the basic recipe
- `5_tune_svm_radial_complex.R`: codes used to tune the SVM radial model using the complex recipe

- `5_train_ensemble_model.R`: codes used to train the ensemble model

- `6_model_analysis.R`: codes used to compare model performance and run times

- `7_tune_refined_bt_basic.R`: codes used train the refined boosted tree model
- `7_tune_refined_rf_basic.R`: codes used train the refined random forest model
- `7_train_refined_ensemble_model.R`: codes used train the refined ensemble model

- `8_refined_model_analysis.R`: codes used to analyze the refined models and the final ensemble model
- `9_assess_final_model.R`: codes used to assess final model and produce predictions using the final model

