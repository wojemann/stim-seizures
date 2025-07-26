# Install packages if needed:
# install.packages("lme4")
# install.packages("lmerTest")
# install.packages("pbkrtest")

library(lme4)
library(lmerTest)  # for Satterthwaite DOF
library(pbkrtest) # for KR DOF

# Set working directory (adjust path as needed)
# setwd("/mnt/sauce/littlab/users/wojemann/stim-seizures")

# Load the data
time_df_all <- read.csv("/mnt/sauce/littlab/users/wojemann/stim-seizures/PROCESSED_DATA/time_df_all.csv")
spread_df_all <- read.csv("/mnt/sauce/littlab/users/wojemann/stim-seizures/PROCESSED_DATA/spread_df_all.csv") 

# Check data structure
cat("=== DATA STRUCTURE ===\n")
cat("Time DF dimensions:", dim(time_df_all), "\n")
cat("Spread DF dimensions:", dim(spread_df_all), "\n")
cat("Null DF dimensions:", dim(null_df_all), "\n")

cat("\nTime DF columns:", colnames(time_df_all), "\n")
cat("Spread DF columns:", colnames(spread_df_all), "\n")
cat("Null DF columns:", colnames(null_df_all), "\n")

# Convert typical to factor
time_df_all$typical <- as.factor(time_df_all$typical)
spread_df_all$typical <- as.factor(spread_df_all$typical)
null_df_all$typical <- as.factor(null_df_all$typical)

# Check levels
cat("\nTypical levels in time_df_all:", levels(time_df_all$typical), "\n")
cat("Typical levels in spread_df_all:", levels(spread_df_all$typical), "\n")

# Summary statistics
cat("\n=== SUMMARY STATISTICS ===\n")
cat("Time DF - Typical distribution:\n")
print(table(time_df_all$typical))

cat("\nSpread DF - Typical distribution:\n") 
print(table(spread_df_all$typical))

cat("\n=== LINEAR MIXED EFFECTS MODELS ===\n")

# Analysis 1: Seizure timing (onset_med_25) comparing Atypical vs Typical
cat("\n--- ANALYSIS 1: SEIZURE TIMING (onset_med_25) ---\n")

# Filter to only Atypical and Typical (exclude "All" if present)
time_df_filtered <- time_df_all[time_df_all$typical %in% c("Atypical", "Typical"), ]
time_df_filtered$typical <- droplevels(time_df_filtered$typical)

# Set Typical as reference level (so coefficient represents Atypical effect)
time_df_filtered$typical <- relevel(time_df_filtered$typical, ref = "Typical")

cat("Sample sizes after filtering:\n")
print(table(time_df_filtered$typical))

# Fit linear mixed effects model: onset_med_25 ~ typical + (1|patient)
timing_model <- lmer(onset_med_25 ~ typical + (1|patient), data = time_df_filtered, REML=FALSE)

cat("\nTiming Model Summary:\n")
print(summary(timing_model))

# Descriptive statistics
cat("\nDescriptive statistics for timing by group:\n")
timing_desc <- aggregate(onset_med_25 ~ typical, data = time_df_filtered, 
                        FUN = function(x) c(mean = mean(x, na.rm = TRUE), 
                                          median = median(x, na.rm = TRUE),
                                          sd = sd(x, na.rm = TRUE),
                                          n = length(x)))
print(timing_desc)

# Analysis 2: Seizure spread (Fraction) comparing Atypical vs Typical  
cat("\n--- ANALYSIS 2: SEIZURE SPREAD (Fraction) ---\n")

# Filter to only Atypical and Typical (exclude "All" if present)
spread_df_filtered <- spread_df_all[spread_df_all$typical %in% c("Atypical", "Typical"), ]
spread_df_filtered$typical <- droplevels(spread_df_filtered$typical)

# Set Typical as reference level
spread_df_filtered$typical <- relevel(spread_df_filtered$typical, ref = "Typical")

cat("Sample sizes after filtering:\n")
print(table(spread_df_filtered$typical))

# Fit linear mixed effects model: Fraction ~ typical + (1|patient)
spread_model <- lmer(Fraction ~ typical + (1|patient), data = spread_df_filtered, REML=FALSE)

cat("\nSpread Model Summary:\n")
print(summary(spread_model))
spread_ols = lm(Fraction ~ typical, data = spread_df_filtered)
print(summary(spread_ols))

# Descriptive statistics
cat("\nDescriptive statistics for spread by group:\n")
spread_desc <- aggregate(Fraction ~ typical, data = spread_df_filtered,
                        FUN = function(x) c(mean = mean(x, na.rm = TRUE),
                                          median = median(x, na.rm = TRUE), 
                                          sd = sd(x, na.rm = TRUE),
                                          n = length(x)))
print(spread_desc)

# Summary of all results
cat("\n=== SUMMARY OF RESULTS ===\n")
cat("All models use random intercepts for patients\n")
cat("All p-values corrected using Satterthwaite degrees of freedom\n\n")

cat("\nAnalysis completed!\n")