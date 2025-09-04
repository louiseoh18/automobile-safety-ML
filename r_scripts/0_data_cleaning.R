# DATA CLEANING & CODEBOOK

# load library
library(tidyverse)
library(here)

#################################################################

# load dataset
crash_data <- read_csv(here("data/automobile_crash.csv")) |>
  janitor::clean_names() |> 
  filter(!is.na(crash_severity)) |> 
  mutate(across(ends_with("_presence"), ~ case_when(. == '1.0' ~ 'True',
                                                    . == '0.0' ~ 'False',
                                                    TRUE ~ as.character(.)))) |> 
  mutate(across(everything(), ~ ifelse(. == 'nan', NA_character_, .))) |> 
  mutate(across(where(is.character), as.factor)) |> 
  mutate(crash_severity = case_when(
    crash_severity == "severe" ~ "severe",
    crash_severity %in% c("minor", "moderate") ~ "not_severe"
    ),
    crash_severity = factor(crash_severity))

# check data
unique(crash_data$crash_severity)
glimpse(crash_data)
skimr::skim(crash_data)

#################################################################

# data codebook
crash_codebook <- as_tibble(data.frame(
  variable = c(
    "vehicle_make", 
    "vehicle_type", 
    "vehicle_year",  
    "engine_type", 
    "engine_displacement",
    "transmission_type",
    "number_of_cylinders",  
    "vehicle_weight", 
    "vehicle_length",
    "vehicle_width",  
    "vehicle_height",  
    "safety_rating",  
    "number_of_airbags",  
    "abs_presence",  
    "esc_presence", 
    "tcs_presence",
    "tpms_presence",
    "crash_location",
    "weather_conditions",
    "road_surface_conditions",
    "time_of_day",
    "day_of_week",
    "driver_age",
    "driver_gender",
    "crash_severity"
  ),
  description = c(
    "the make or manufacturer of the vehicle involved in the crash", 
    "the type of vehicle involved in the crash",
    "the year of manufacture for the vehicle",  
    "the type of engine powering the vehicle",
    "the displacement of the engine in cubic centimeters (cc)",
    "indicates whether the vehicle has a manual or automatic transmission",
    "the number of cylinders in the vehicle's engine",  
    "the weight of the vehicle in kilograms (kg)", 
    "the length of the vehicle in millimeters (mm)",
    "the width of the vehicle in millimeters (mm)",  
    "the height of the vehicle in millimeters (mm)",  
    "assigns a safety rating to the vehicle based on predefined criteria",
    "specifies the number of airbags installed in the vehicle",
    "indicates whether the vehicle is equipped with an anti-lock braking system (abs)",
    "indicates whether the vehicle is equipped with electronic stability control (esc)",
    "indicates whether the vehicle is equipped with traction control system (tcs)",
    "indicates whether the vehicle is equipped with a tire pressure monitoring system (tpms)",
    "specifies the location of the crash as urban or rural",
    "describes prevailing weather conditions during the crash",
    "indicates the condition of the road surface at the time of the crash",
    "specifies the time of day when the crash occurred",
    "indicates the day of the week when the crash occurred",
    "the age of the driver involved in the crash",
    "the gender of the driver involved in the crash",
    "classifies the severity of the crash as severe or not severe"
  )
))

file_path_1 <- file.path("data", "automobile_crash_codebook.csv")
write_csv(crash_codebook, file = file_path_1)

#################################################################

# save files
save(crash_data, file = here("data/automobile_crash_clean.rda"))

