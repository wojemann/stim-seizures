# Install packages if needed:
# install.packages("lme4")
# install.packages("lmerTest")
# install.packages("pbkrtest")

library(lme4)
library(lmerTest)  # for Satterthwaite DOF
library(pbkrtest) # for KR DOF

# Read the data
# Main analysis uses modeling_df_75.csv
df <- read.csv("/Users/wojemann/Documents/CNT/stim_seizures_data/PROCESSED_DATA/modeling_df_75.csv")
# Outcome analysis uses modeling_df_25.csv (loaded separately below)

# Binarize age, duration, and age_at_onset at their medians
cat("\n--- Binarizing age, duration, and age_at_onset at their medians ---\n")
df$age_bin <- as.integer(df$age > median(df$age, na.rm=TRUE))
df$duration_bin <- as.integer(df$duration > median(df$duration, na.rm=TRUE))
df$age_at_onset_bin <- as.integer(df$age_at_onset > median(df$age_at_onset, na.rm=TRUE))

# 1. Typical vs. Atypical Stim Seizures (KR-corrected p-value)
cat("\n--- Typical vs. Atypical Stim Seizures (Satterthwaite-corrected p-value) ---\n")
cat("Fixed effects: Regression coefficients (Estimate), standard errors, Kenward-Roger degrees of freedom, t-values, and p-values for each predictor.\n")
model_typical <- lmer(MCC ~ typical + (1|patient), data = df, REML=FALSE)
print(summary(model_typical))
cat("Random effect variance: Variance attributed to the random intercept for patient, and residual variance.\n")
# print(VarCorr(model_typical),comp="Variance")

# One-sided test: Typical > Atypical (higher MCC for typical)
cat("\n--- One-sided test: Typical > Atypical (higher MCC for typical) ---\n")
coefs_typical <- coef(summary(model_typical))
# Check for typical coefficient (could be typicalTRUE, typicalTrue, typical, or typical1 depending on coding)
typical_coef_name <- NULL
# First try exact matches
for (name in c("typicalTRUE", "typicalTrue", "typical", "typical1")) {
  if (name %in% rownames(coefs_typical)) {
    typical_coef_name <- name
    break
  }
}
# If not found, search for any coefficient starting with "typical" (case-insensitive)
if (is.null(typical_coef_name)) {
  typical_matches <- grep("^typical", rownames(coefs_typical), ignore.case = TRUE, value = TRUE)
  if (length(typical_matches) > 0) {
    typical_coef_name <- typical_matches[1]
  }
}
if (!is.null(typical_coef_name)) {
  t_stat_typical <- coefs_typical[typical_coef_name, "t value"]
  df_typical <- coefs_typical[typical_coef_name, "df"]
  # One-sided p-value: P(T > t_stat) if t_stat > 0, else P(T < t_stat)
  if (t_stat_typical > 0) {
    p_one_sided_typical <- pt(t_stat_typical, df = df_typical, lower.tail = FALSE)
    cat(sprintf("One-sided test (H1: typical > atypical): coefficient = %s, t = %.4f, df = %.2f, p = %.4g\n", 
                typical_coef_name, t_stat_typical, df_typical, p_one_sided_typical))
  } else {
    p_one_sided_typical <- pt(t_stat_typical, df = df_typical, lower.tail = TRUE)
    cat(sprintf("One-sided test (H1: typical > atypical): coefficient = %s, t = %.4f, df = %.2f, p = %.4g (note: t-stat is negative, so p > 0.5)\n", 
                typical_coef_name, t_stat_typical, df_typical, p_one_sided_typical))
  }
} else {
  cat("Warning: 'typical' coefficient not found in model summary. Available coefficients:\n")
  print(rownames(coefs_typical))
}

# 2. Typical vs. Typical x Center (KR LRT)
cat("\n--- Typical vs. Typical x Center (KR LRT) ---\n")
cat("KR LRT: F-test statistic, numerator and denominator degrees of freedom (Kenward-Roger), scaling factor, and p-value for the interaction between typical and center.\n")
model_typical_center <- lmer(MCC ~ typical * adult + (1|patient), data = df, REML=FALSE)
print(summary(model_typical_center))
kr_test_interaction <- KRmodcomp(model_typical_center, model_typical)
print(kr_test_interaction)

# 3. LME Coefficients for various models (Standard t-tests)
# a. MTLE
cat("\n--- MTLE Model ---\n")
cat("Fixed effects: Regression coefficients, standard errors, Satterthwaite degrees of freedom, t-values, and p-values for each predictor.\n") # nolint
model_mtle <- lmer(MCC ~ mtle + (1|patient), data = df, REML=FALSE)
print(summary(model_mtle)$coefficients)
cat("Random effect variance: Variance attributed to the random intercept for patient, and residual variance.\n") # nolint
print(VarCorr(model_mtle),comp="Variance")


# b. Center x MTLE
cat("\n--- Adult x MTLE Model ---\n")
cat("Fixed effects: Regression coefficients, standard errors, Satterthwaite degrees of freedom, t-values, and p-values for each predictor.\n") # nolint
model_center_mtle <- lmer(MCC ~ adult * mtle + (1|patient), data = df, REML=FALSE)
print(summary(model_center_mtle)$coefficients)
cat("Random effect variance: Variance attributed to the random intercept for patient, and residual variance.\n") # nolint
print(VarCorr(model_center_mtle),comp="Variance")

# --- Alternative contrast for Adult x MTLE (LME) ---
cat("\n--- Alternative contrast: adultTrue + adultTrue:mtleTrue (LME) ---\n")
# Extract coefficients and covariance matrix
coefs_adult_mtle_lme <- coef(summary(model_center_mtle))
cov_adult_mtle_lme <- vcov(model_center_mtle)
# Estimate: sum of main effect and interaction
estimate_adult_mtle_lme <- fixef(model_center_mtle)["mtleTrue"] + fixef(model_center_mtle)["adultTrue:mtleTrue"]
# Variance: var1 + var3 + 2*cov13
var1_adult_mtle_lme <- cov_adult_mtle_lme["mtleTrue", "mtleTrue"]
var3_adult_mtle_lme <- cov_adult_mtle_lme["adultTrue:mtleTrue", "adultTrue:mtleTrue"]
cov13_adult_mtle_lme <- cov_adult_mtle_lme["mtleTrue", "adultTrue:mtleTrue"]
se_adult_mtle_lme <- sqrt(var1_adult_mtle_lme + var3_adult_mtle_lme + 2 * cov13_adult_mtle_lme)
t_stat_adult_mtle_lme <- estimate_adult_mtle_lme / se_adult_mtle_lme
# Use conservative df (minimum of the two coefficients' df)
df_adult_mtle_lme <- min(coefs_adult_mtle_lme["mtleTrue", "df"], coefs_adult_mtle_lme["adultTrue:mtleTrue", "df"])
p_value_adult_mtle_lme <- 2 * pt(-abs(t_stat_adult_mtle_lme), df = df_adult_mtle_lme)
cat(sprintf("beta: %.4f, se: %.4f, t-stat: %.4f, df: %.2f, p-value: %.4g\n", estimate_adult_mtle_lme, se_adult_mtle_lme, t_stat_adult_mtle_lme, df_adult_mtle_lme, p_value_adult_mtle_lme))

center_ps_lme = c(coefs_adult_mtle_lme["mtleTrue","Pr(>|t|)"], coefs_adult_mtle_lme["adultTrue:mtleTrue","Pr(>|t|)"], p_value_adult_mtle_lme)
cat("Adult x MTLE Model (LME) Bonferroni-adjusted p-values:")
cat("\n","mtleTrue","adultTrue:mtleTrue","p_value_adult_mtle","\n")
print(p.adjust(center_ps_lme, method="bonferroni"))

# c. Duration x MTLE (binarized)
cat("\n--- Duration_bin x MTLE Model ---\n")
cat("Fixed effects: Regression coefficients, standard errors, Satterthwaite degrees of freedom, t-values, and p-values for each predictor.\n") # nolint
model_durationbin_mtle <- lmer(MCC ~ duration_bin * mtle + (1|patient), data = df, REML=FALSE)
print(summary(model_durationbin_mtle)$coefficients)
cat("Random effect variance: Variance attributed to the random intercept for patient, and residual variance.\n") # nolint
print(VarCorr(model_durationbin_mtle),comp="Variance")

# d. Age x MTLE (binarized)
cat("\n--- Age_bin x MTLE Model ---\n")
cat("Fixed effects: Regression coefficients, standard errors, Satterthwaite degrees of freedom, t-values, and p-values for each predictor.\n") # nolint
model_agebin_mtle <- lmer(MCC ~ age_bin * mtle + (1|patient), data = df, REML=FALSE)
print(summary(model_agebin_mtle)$coefficients)
cat("Random effect variance: Variance attributed to the random intercept for patient, and residual variance.\n") # nolint
print(VarCorr(model_agebin_mtle),comp="Variance")

# e. Age at Onset x MTLE (binarized and unbinarized)
cat("\n--- Age at Onset_bin x MTLE Model ---\n")
cat("Fixed effects: Regression coefficients, standard errors, Satterthwaite degrees of freedom, t-values, and p-values for each predictor.\n") # nolint
model_ageonsetbin_mtle <- lmer(MCC ~ age_at_onset_bin * mtle + (1|patient), data = df, REML=FALSE)
print(summary(model_ageonsetbin_mtle)$coefficients)
cat("Random effect variance: Variance attributed to the random intercept for patient, and residual variance.\n") # nolint
print(VarCorr(model_ageonsetbin_mtle),comp="Variance")

# --- Alternative contrast for Age at Onset_bin x MTLE (LME) ---
cat("\n--- Alternative contrast: age_at_onset_bin + age_at_onset_bin:mtleTrue (LME) ---\n")
coefs_onset_mtle_lme <- coef(summary(model_ageonsetbin_mtle))
cov_onset_mtle_lme <- vcov(model_ageonsetbin_mtle)
estimate_onset_mtle_lme <- fixef(model_ageonsetbin_mtle)["mtleTrue"] + fixef(model_ageonsetbin_mtle)["age_at_onset_bin:mtleTrue"]
var1_onset_mtle_lme <- cov_onset_mtle_lme["mtleTrue", "mtleTrue"]
var3_onset_mtle_lme <- cov_onset_mtle_lme["age_at_onset_bin:mtleTrue", "age_at_onset_bin:mtleTrue"]
cov13_onset_mtle_lme <- cov_onset_mtle_lme["mtleTrue", "age_at_onset_bin:mtleTrue"]
se_onset_mtle_lme <- sqrt(var1_onset_mtle_lme + var3_onset_mtle_lme + 2 * cov13_onset_mtle_lme)
t_stat_onset_mtle_lme <- estimate_onset_mtle_lme / se_onset_mtle_lme
df_onset_mtle_lme <- min(coefs_onset_mtle_lme["mtleTrue", "df"], coefs_onset_mtle_lme["age_at_onset_bin:mtleTrue", "df"])
p_value_onset_mtle_lme <- 2 * pt(-abs(t_stat_onset_mtle_lme), df = df_onset_mtle_lme)
cat(sprintf("beta: %.4f, se: %.4f, t-stat: %.4f, df: %.2f, p-value: %.4g\n", estimate_onset_mtle_lme, se_onset_mtle_lme, t_stat_onset_mtle_lme, df_onset_mtle_lme, p_value_onset_mtle_lme))
age_ps_lme = c(coefs_onset_mtle_lme["mtleTrue","Pr(>|t|)"], coefs_onset_mtle_lme["age_at_onset_bin:mtleTrue","Pr(>|t|)"], p_value_onset_mtle_lme)
cat("Age at Onset_bin x MTLE Model (LME) Bonferroni-adjusted p-values:")
cat("\n","mtleTrue","age_at_onset_bin:mtleTrue","p_value_onset_mtle","\n")
print(p.adjust(age_ps_lme, method="bonferroni"))

# ===== OLS ANALYSES (NO RANDOM EFFECTS) =====
cat("\n\n========== OLS ANALYSES (NO RANDOM EFFECTS) ==========\n")

# OLS Coefficients for various models (Standard t-tests)
# a. MTLE
cat("\n--- MTLE Model (OLS) ---\n")
cat("OLS coefficients: Regression coefficients, standard errors, t-values, and p-values for each predictor (no random effects).\n") # nolint
model_mtle_ols <- lm(MCC ~ mtle, data = df)
print(summary(model_mtle_ols))

# b. Adult x MTLE
cat("\n--- Adult x MTLE Model (OLS) ---\n")
cat("OLS coefficients: Regression coefficients, standard errors, t-values, and p-values for each predictor (no random effects).\n") # nolint
model_adult_mtle_ols <- lm(MCC ~ adult * mtle, data = df)
print(summary(model_adult_mtle_ols))

# c. Duration x MTLE (binarized)
cat("\n--- Duration_bin x MTLE Model (OLS) ---\n")
cat("OLS coefficients: Regression coefficients, standard errors, t-values, and p-values for each predictor (no random effects).\n") # nolint
model_durationbin_mtle_ols <- lm(MCC ~ duration_bin * mtle, data = df)
print(summary(model_durationbin_mtle_ols))

# d. Age x MTLE (binarized)
cat("\n--- Age_bin x MTLE Model (OLS) ---\n")
cat("OLS coefficients: Regression coefficients, standard errors, t-values, and p-values for each predictor (no random effects).\n") # nolint
model_agebin_mtle_ols <- lm(MCC ~ age_bin * mtle, data = df)
print(summary(model_agebin_mtle_ols))

# e. Age at Onset x MTLE (binarized)
cat("\n--- Age at Onset_bin x MTLE Model (OLS) ---\n")
cat("OLS coefficients: Regression coefficients, standard errors, t-values, and p-values for each predictor (no random effects).\n") # nolint
vars_needed <- c("MCC", "mtle", "age_at_onset_bin", "patient")
df_complete <- df[complete.cases(df[, vars_needed]), ]

model_mtle_cc <- lm(MCC ~ mtle, data = df_complete)
model_ageonsetbin_mtle_ols <- lm(MCC ~ age_at_onset_bin * mtle, data = df_complete)
print(summary(model_ageonsetbin_mtle_ols))


# --- Alternative contrast for Adult x MTLE (OLS) ---
cat("\n--- Alternative contrast: adultTrue + adultTrue:mtleTrue (OLS) ---\n")
# Extract coefficients and covariance matrix
coefs_adult_mtle <- coef(summary(model_adult_mtle_ols))
cov_adult_mtle <- vcov(model_adult_mtle_ols)
# Estimate: sum of main effect and interaction
estimate_adult_mtle <- coef(model_adult_mtle_ols)["mtleTrue"] + coef(model_adult_mtle_ols)["adultTrue:mtleTrue"]
# Variance: var1 + var3 + 2*cov13
var1_adult_mtle <- cov_adult_mtle["mtleTrue", "mtleTrue"]
var3_adult_mtle <- cov_adult_mtle["adultTrue:mtleTrue", "adultTrue:mtleTrue"]
cov13_adult_mtle <- cov_adult_mtle["mtleTrue", "adultTrue:mtleTrue"]
se_adult_mtle <- sqrt(var1_adult_mtle + var3_adult_mtle + 2 * cov13_adult_mtle)
t_stat_adult_mtle <- estimate_adult_mtle / se_adult_mtle
df_adult_mtle <- model_adult_mtle_ols$df.residual
p_value_adult_mtle <- 2 * pt(-abs(t_stat_adult_mtle), df = df_adult_mtle)
cat(sprintf("beta: %.4f, se: %.4f, t-stat: %.4f, p-value: %.4g\n", estimate_adult_mtle, se_adult_mtle, t_stat_adult_mtle, p_value_adult_mtle))

center_ps = c(coefs_adult_mtle["mtleTrue","Pr(>|t|)"],coefs_adult_mtle["adultTrue:mtleTrue","Pr(>|t|)"],p_value_adult_mtle)
cat("Adult x MTLE Model (OLS) Bonferroni-adjusted p-values:")
cat("\n","mtleTrue","adultTrue:mtleTrue","p_value_adult_mtle","\n")
print(p.adjust(center_ps,method="bonferroni"))

# --- Alternative contrast for Age at Onset_bin x MTLE (OLS) ---
cat("\n--- Alternative contrast: age_at_onset_bin + age_at_onset_bin:mtleTrue (OLS) ---\n")
coefs_onset_mtle <- coef(summary(model_ageonsetbin_mtle_ols))
cov_onset_mtle <- vcov(model_ageonsetbin_mtle_ols)
estimate_onset_mtle <- coef(model_ageonsetbin_mtle_ols)["mtleTrue"] + coef(model_ageonsetbin_mtle_ols)["age_at_onset_bin:mtleTrue"]
var1_onset_mtle <- cov_onset_mtle["mtleTrue", "mtleTrue"]
var3_onset_mtle <- cov_onset_mtle["age_at_onset_bin:mtleTrue", "age_at_onset_bin:mtleTrue"]
cov13_onset_mtle <- cov_onset_mtle["mtleTrue", "age_at_onset_bin:mtleTrue"]
se_onset_mtle <- sqrt(var1_onset_mtle + var3_onset_mtle + 2 * cov13_onset_mtle)
t_stat_onset_mtle <- estimate_onset_mtle / se_onset_mtle
df_onset_mtle <- model_ageonsetbin_mtle_ols$df.residual
p_value_onset_mtle <- 2 * pt(-abs(t_stat_onset_mtle), df = df_onset_mtle)
cat(sprintf("beta: %.4f, se: %.4f, t-stat: %.4f, p-value: %.4g\n", estimate_onset_mtle, se_onset_mtle, t_stat_onset_mtle, p_value_onset_mtle))
age_ps = c(coefs_onset_mtle["mtleTrue","Pr(>|t|)"],coefs_onset_mtle["age_at_onset_bin:mtleTrue","Pr(>|t|)"],p_value_onset_mtle)
cat("Age at Onset_bin x MTLE Model (OLS) Bonferroni-adjusted p-values:")
cat("\n","mtleTrue","age_at_onset_bin:mtleTrue","p_value_onset_mtle","\n")
print(p.adjust(age_ps,method="bonferroni"))

# ===== F-TESTS FOR OLS MODEL COMPARISONS =====

# Compare adult*mtle interaction model to mtle-only model
cat("\n--- F-test: Adult x MTLE vs. MTLE-only Model (OLS) ---\n")
cat("F-test: Compares nested OLS models. Shows residual degrees of freedom, residual sum of squares, difference in degrees of freedom, difference in sum of squares, F statistic, and p-value.\n") # nolint
f_test_adult_mtle <- anova(model_mtle_ols, model_adult_mtle_ols)
print(f_test_adult_mtle)

# Compare age_at_onset_bin*mtle interaction model to mtle-only model
cat("\n--- F-test: Age at Onset_bin x MTLE vs. MTLE-only Model (OLS) ---\n")
cat("F-test: Compares nested OLS models. Shows residual degrees of freedom, residual sum of squares, difference in degrees of freedom, difference in sum of squares, F statistic, and p-value.\n") # nolint
f_test_onset_mtle <- anova(model_mtle_cc, model_ageonsetbin_mtle_ols)
print(f_test_onset_mtle)

# ===== OLS MODEL FIT STATISTICS =====

# Model fit statistics for age-based and duration-based models
cat("\n--- Model Fit Statistics for OLS Models ---\n")
cat("For each OLS model: AIC (Akaike Information Criterion), BIC (Bayesian Information Criterion), R-squared (proportion of variance explained), and adjusted R-squared.\n") # nolint 
age_duration_models_ols <- list(
  mtle_only = model_mtle_ols,
  age_bin_x_mtle = model_agebin_mtle_ols,
  age_at_onset_bin_x_mtle = model_ageonsetbin_mtle_ols,
  duration_bin_x_mtle = model_durationbin_mtle_ols,
  adult_x_mtle = model_adult_mtle_ols
)

for (name in names(age_duration_models_ols)) {
  m <- age_duration_models_ols[[name]]
  cat(sprintf("\nModel: %s\n", name))
  cat(sprintf("  AIC: %.2f\n", AIC(m)))
  cat(sprintf("  BIC: %.2f\n", BIC(m)))
  cat(sprintf("  R-squared: %.3f\n", summary(m)$r.squared))
  cat(sprintf("  Adj R-squared: %.3f\n", summary(m)$adj.r.squared))
}

# ===== OUTCOME ANALYSIS (SEPARATE SECTION) =====
cat("\n\n========== OUTCOME ANALYSIS ==========\n")

# Load outcome data from modeling_df_25.csv
df_outcome <- read.csv("/Users/wojemann/Documents/CNT/stim_seizures_data/PROCESSED_DATA/modeling_df_25.csv")

cat("\n--- Outcome Model (LME) ---\n")
cat("Fixed effects: Regression coefficients, standard errors, Satterthwaite degrees of freedom, t-values, and p-values for each predictor.\n") # nolint
# Drop rows with missing outcome values
vars_needed_outcome <- c("MCC", "outcome", "patient")
df_outcome_complete <- df_outcome[complete.cases(df_outcome[, vars_needed_outcome]), ]

cat(sprintf("Number of rows: %d\n", nrow(df_outcome_complete)))
cat(sprintf("Number of patients: %d\n", length(unique(df_outcome_complete$patient))))

# Use mixed-effects model to account for repeated measures from patients
model_outcome <- lmer(MCC ~ outcome + (1|patient), data = df_outcome_complete, REML=TRUE)
print(summary(model_outcome))
cat("Random effect variance: Variance attributed to the random intercept for patient, and residual variance.\n") # nolint
print(VarCorr(model_outcome),comp="Variance")

# One-sided test: Good outcome (1) > Poor outcome (0) (higher MCC for good outcome)
cat("\n--- One-sided test: Good outcome (1) > Poor outcome (0) (higher MCC for good outcome) ---\n")
coefs_outcome <- coef(summary(model_outcome))
# Check for outcome coefficient (could be outcome, outcome1, outcomeTRUE, or outcomeTrue depending on coding)
outcome_coef_name <- NULL
# First try exact matches
for (name in c("outcome", "outcome1", "outcomeTRUE", "outcomeTrue")) {
  if (name %in% rownames(coefs_outcome)) {
    outcome_coef_name <- name
    break
  }
}
# If not found, search for any coefficient starting with "outcome" (case-insensitive)
if (is.null(outcome_coef_name)) {
  outcome_matches <- grep("^outcome", rownames(coefs_outcome), ignore.case = TRUE, value = TRUE)
  if (length(outcome_matches) > 0) {
    outcome_coef_name <- outcome_matches[1]
  }
}
if (!is.null(outcome_coef_name)) {
  t_stat_outcome <- coefs_outcome[outcome_coef_name, "t value"]
  df_outcome <- coefs_outcome[outcome_coef_name, "df"]
  # One-sided p-value: P(T > t_stat) if t_stat > 0, else P(T < t_stat)
  if (t_stat_outcome > 0) {
    p_one_sided_outcome <- pt(t_stat_outcome, df = df_outcome, lower.tail = FALSE)
    cat(sprintf("One-sided test (H1: outcome > 0, i.e., good outcome > poor outcome): coefficient = %s, t = %.4f, df = %.2f, p = %.4g\n", 
                outcome_coef_name, t_stat_outcome, df_outcome, p_one_sided_outcome))
  } else {
    p_one_sided_outcome <- pt(t_stat_outcome, df = df_outcome, lower.tail = TRUE)
    cat(sprintf("One-sided test (H1: outcome > 0, i.e., good outcome > poor outcome): coefficient = %s, t = %.4f, df = %.2f, p = %.4g (note: t-stat is negative, so p > 0.5)\n", 
                outcome_coef_name, t_stat_outcome, df_outcome, p_one_sided_outcome))
  }
} else {
  cat("Warning: 'outcome' coefficient not found in model summary. Available coefficients:\n")
  print(rownames(coefs_outcome))
}
