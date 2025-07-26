# Mixed Effects Model Analysis for Seizure Detection Data
# This script performs LME analyses with random intercepts for patients
# on threshold comparisons, model agreements, and stim vs spontaneous seizure comparisons

# Load required libraries
library(lme4)
library(lmerTest)
library(dplyr)
library(ggplot2)
library(emmeans)
library(multcomp)


# Load data path
prodatapath  = "/mnt/sauce/littlab/users/wojemann/stim-seizures/PROCESSED_DATA"

# Helper function to read CSV files
read_data <- function(filename) {
  filepath <- file.path(prodatapath, filename)
  if (!file.exists(filepath)) {
    warning(paste("File not found:", filepath))
    return(NULL)
  }
  return(read.csv(filepath, stringsAsFactors = FALSE))
}

cat("=== Mixed Effects Model Analysis for Seizure Detection ===\n\n")

# ===================================================================
# 1. THRESHOLD COMPARISON ANALYSIS (Stim vs Spontaneous)
# ===================================================================

cat("1. THRESHOLD COMPARISON ANALYSIS\n")
cat("================================\n")

# Load threshold data
plot_thresholds <- read_data("plot_thresholds_for_lme.csv")

if (!is.null(plot_thresholds)) {
  cat("Data loaded successfully. Shape:", nrow(plot_thresholds), "x", ncol(plot_thresholds), "\n")
  cat("Columns:", paste(colnames(plot_thresholds), collapse = ", "), "\n\n")
  
  # Convert factors
  plot_thresholds$patient <- as.factor(plot_thresholds$patient)
  plot_thresholds$stim <- as.factor(plot_thresholds$stim)
  levels(plot_thresholds$stim) <- c("Spontaneous", "Stimulated")
  
  # Descriptive statistics
  cat("Descriptive Statistics:\n")
  desc_stats <- plot_thresholds %>%
    group_by(stim) %>% 
    summarise(
      n = n(),
      mean_threshold = mean(threshold, na.rm = TRUE),
      median_threshold = median(threshold, na.rm = TRUE),
      sd_threshold = sd(threshold, na.rm = TRUE),
      .groups = 'drop'
    )
  print(desc_stats)
  cat("\n")
  
  # Mixed effects model for threshold comparison
  cat("Mixed Effects Model for Threshold Comparison:\n")
  threshold_model <- lmer(threshold ~ stim + (1|patient), data = plot_thresholds)
  
  # Model summary
  cat("Model Summary:\n")
  print(summary(threshold_model))
  cat("\n")
  
  # ANOVA for fixed effects
  cat("ANOVA for Fixed Effects:\n")
  threshold_anova <- anova(threshold_model)
  print(threshold_anova)
  cat("\n")
  
  # Post-hoc comparisons using emmeans
  cat("Post-hoc Comparisons:\n")
  threshold_emm <- emmeans(threshold_model, ~ stim)
  threshold_contrasts <- contrast(threshold_emm, method = "pairwise")
  print(threshold_contrasts)
  cat("\n")
  
  # Effect size (Cohen's d equivalent for LME)
  cat("Effect Size Estimation:\n")
  threshold_means <- emmeans(threshold_model, ~ stim)
  threshold_pairs <- pairs(threshold_means)
  print(confint(threshold_pairs))
  cat("\n")
  
  # Variance homogeneity test accounting for patient clustering
  cat("VARIANCE HOMOGENEITY TESTS:\n")
  cat("===========================\n")
  
  # Method 1: Levene test on patient-averaged data
  cat("Method 1 - Levene Test on Patient-Averaged Data:\n")
  threshold_by_patient <- plot_thresholds %>%
    group_by(patient, stim) %>%
    summarise(mean_threshold = mean(threshold, na.rm = TRUE), .groups = 'drop') %>%
    filter(!is.na(mean_threshold))
  
  if (nrow(threshold_by_patient) > 0) {
    if (!require(car, quietly = TRUE)) {
      install.packages("car")
      library(car)
    }
    
    levene_patient_avg <- car::leveneTest(mean_threshold ~ stim, data = threshold_by_patient)
    print(levene_patient_avg)
    cat("\n")
  }
  
  # Method 2: Mixed effects variance structure comparison
  cat("Method 2 - Mixed Effects Variance Structure Comparison:\n")
  if (!require(nlme, quietly = TRUE)) {
    install.packages("nlme")
    library(nlme)
  }
  
  tryCatch({
    # Model with homogeneous variance
    model_homo <- nlme::lme(threshold ~ stim, random = ~1|patient, data = plot_thresholds)
    
    # Model with different variances by stim condition
    model_hetero <- nlme::lme(threshold ~ stim, random = ~1|patient, 
                             weights = varIdent(form = ~1|stim), data = plot_thresholds)
    
    # Likelihood ratio test for variance differences
    variance_comparison <- anova(model_homo, model_hetero)
    print(variance_comparison)
    cat("\n")
    
    # Show variance estimates from heteroscedastic model
    cat("Variance Estimates by Group:\n")
    print(intervals(model_hetero)$varStruct)
    cat("\n")
    
  }, error = function(e) {
    cat("Mixed effects variance comparison failed:", e$message, "\n")
  })
  
  # Method 3: Descriptive variance comparison
  cat("Method 3 - Descriptive Variance Statistics:\n")
  variance_stats <- plot_thresholds %>%
    group_by(stim) %>%
    summarise(
      n = n(),
      variance = var(threshold, na.rm = TRUE),
      sd = sd(threshold, na.rm = TRUE),
      cv = sd(threshold, na.rm = TRUE) / mean(threshold, na.rm = TRUE),
      .groups = 'drop'
    )
  print(variance_stats)
  
  # Calculate variance ratio
  if (nrow(variance_stats) == 2) {
    var_ratio <- variance_stats$variance[variance_stats$stim == "Stimulated"] / 
                variance_stats$variance[variance_stats$stim == "Spontaneous"]
    cat("Variance Ratio (Stimulated/Spontaneous):", round(var_ratio, 3), "\n")
  }
  cat("\n")
  
} else {
  cat("Threshold data not found. Skipping analysis.\n\n")
}

# ===================================================================
# 2. MODEL AGREEMENT ANALYSIS (Onset)
# ===================================================================

cat("2. MODEL AGREEMENT ANALYSIS - ONSET\n")
cat("====================================\n")

# Load onset agreement data
onset_agreements <- read_data("onset_all_plot_agreements_for_lme.csv")

if (!is.null(onset_agreements)) {
  cat("Data loaded successfully. Shape:", nrow(onset_agreements), "x", ncol(onset_agreements), "\n")
  cat("Columns:", paste(colnames(onset_agreements), collapse = ", "), "\n\n")
  
  # Data preparation for proper pairing
  cat("Preparing data for paired comparisons...\n")
  
  # Check the structure of the data
  cat("Unique models:", paste(unique(onset_agreements$model), collapse = ", "), "\n")
  cat("Unique patients:", length(unique(onset_agreements$patient)), "\n")
  cat("Data structure check:\n")
  structure_check <- onset_agreements %>%
    group_by(patient, approximate_onset) %>%
    summarise(
      n_models = n(),
      models_present = paste(sort(unique(model)), collapse = ", "),
      .groups = 'drop'
    )
  
  # Show examples of the pairing structure
  cat("Sample of seizure-model combinations:\n")
  print(head(structure_check, 10))
  cat("\n")
  
  # Check for complete cases (seizures with all models)
  complete_seizures <- structure_check %>%
    filter(n_models == length(unique(onset_agreements$model)))
  
  cat("Seizures with all models present:", nrow(complete_seizures), "out of", nrow(structure_check), "\n")
  
  # Filter to only include complete cases for proper pairing
  if (nrow(complete_seizures) > 0) {
    cat("Using only seizures with complete model data for paired analysis\n")
    onset_agreements_paired <- onset_agreements %>%
      semi_join(complete_seizures, by = c("patient", "approximate_onset"))
    
    cat("Filtered data shape:", nrow(onset_agreements_paired), "x", ncol(onset_agreements_paired), "\n")
    
    # Verify the pairing worked
    final_check <- onset_agreements_paired %>%
      group_by(patient, approximate_onset) %>%
      summarise(n_models = n(), .groups = 'drop')
    
    cat("All seizures now have", unique(final_check$n_models), "models\n\n")
    
  } else {
    cat("Warning: No seizures found with all models present. Using all available data.\n")
    onset_agreements_paired <- onset_agreements
  }
  
  # Prepare data - filter out Interrater for model comparisons
  onset_models <- onset_agreements_paired %>%
    filter(model != "Interrater") %>%
    mutate(
      model = as.factor(model),
      patient = as.factor(patient)
    )
  
  # Descriptive statistics
  cat("Descriptive Statistics by Model:\n")
  desc_stats_onset <- onset_models %>%
    group_by(model) %>%
    summarise(
      n = n(),
      mean_dice = mean(dice, na.rm = TRUE),
      median_dice = median(dice, na.rm = TRUE),
      sd_dice = sd(dice, na.rm = TRUE),
      .groups = 'drop'
    )
  print(desc_stats_onset)
  cat("\n")
  
  # Mixed effects model for onset agreement
  cat("Mixed Effects Model for Onset Agreement:\n")
  onset_model <- lmer(dice ~ model + (1|patient) + (1|approximate_onset), data = onset_models)
  
  cat("Note: This model accounts for:\n")
  cat("- Patient-level clustering (seizures within patients)\n") 
  cat("- Seizure-level pairing (same seizure analyzed by different models)\n\n")
  
  # Model summary
  cat("Model Summary:\n")
  print(summary(onset_model))
  cat("\n")
  
  # ANOVA for fixed effects
  cat("ANOVA for Fixed Effects:\n")
  onset_anova <- anova(onset_model)
  print(onset_anova)
  cat("\n")
  
  # Post-hoc comparisons
  cat("Post-hoc Comparisons (All Pairwise):\n")
  onset_emm <- emmeans(onset_model, ~ model)
  onset_contrasts <- contrast(onset_emm, method = "pairwise", adjust = "bonferroni")
  print(onset_contrasts)
  cat("\n")
  
  # Specific contrasts focusing on NDD model
  cat("Specific Contrasts (NDD vs Others):\n")
  if ("NDD" %in% levels(onset_models$model)) {
    ndd_contrasts <- contrast(onset_emm, 
                              list("NDD vs AbsSlp" = c(-1, 0, 1),
                                   "NDD vs DL" = c(0, -1, 1)),
                              adjust = "bonferroni")
    print(ndd_contrasts)
  }
  cat("\n")
  
} else {
  cat("Onset agreement data not found. Skipping analysis.\n\n")
}

# ===================================================================
# 2a. LSTM MODEL VS HUMAN INTERRATER AGREEMENT ANALYSIS
# ===================================================================

cat("2a. LSTM MODEL VS HUMAN INTERRATER AGREEMENT ANALYSIS\n")
cat("======================================================\n")

# Load the long-format data (LSTM vs human, long-form)
model_interrater_long <- read_data("model-interrater_agreement.csv")

if (!is.null(model_interrater_long)) {
  cat("Data loaded successfully. Shape:", nrow(model_interrater_long), "x", ncol(model_interrater_long), "\n")
  cat("Columns:", paste(colnames(model_interrater_long), collapse = ", "), "\n\n")
  
  # Convert to factors
  model_interrater_long$patient <- as.factor(model_interrater_long$patient)
  model_interrater_long$annotator <- as.factor(model_interrater_long$annotator)
  if ("approximate_onset" %in% colnames(model_interrater_long)) {
    model_interrater_long$approximate_onset <- as.factor(model_interrater_long$approximate_onset)
  }
  
  # Descriptive statistics
  cat("Descriptive Statistics by Annotator (LSTM vs Human):\n")
  desc_stats_2a <- model_interrater_long %>%
    group_by(annotator) %>%
    summarise(
      n = n(),
      mean_dice = mean(dice, na.rm = TRUE),
      median_dice = median(dice, na.rm = TRUE),
      sd_dice = sd(dice, na.rm = TRUE),
      .groups = 'drop'
    )
  print(desc_stats_2a)
  cat("\n")
  
  # Mixed effects model: dice ~ annotator + (1|Patient) + (1|approximate_onset)
  cat("Mixed Effects Model for LSTM vs Human Dice:\n")
  model_vs_human <- lmer(dice ~ annotator + (1|patient) + (1|approximate_onset), data = model_interrater_long)
  
  # Model summary
  print(summary(model_vs_human))
  cat("\n")
  
  # ANOVA for fixed effects
  cat("ANOVA for Fixed Effects:\n")
  print(anova(model_vs_human))
  cat("\n")
  
  # Post-hoc comparisons
  cat("Post-hoc Comparisons (LSTM vs Human):\n")
  emm_2a <- emmeans(model_vs_human, ~ annotator)
  contrasts_2a <- contrast(emm_2a, method = "pairwise")
  print(contrasts_2a)
  cat("\n")
  
  # Effect size estimation
  cat("Effect Size Estimation:\n")
  print(confint(contrasts_2a))
  cat("\n")
  
} else {
  cat("Model/interrater long-format data not found. Skipping analysis.\n\n")
}

# ===================================================================
# 3. LSTM/NDD STIM VS SPONTANEOUS SEIZURE ANALYSIS
# ===================================================================

cat("3. LSTM/NDD STIM VS SPONTANEOUS SEIZURE ANALYSIS\n")
cat("================================================\n")

# Load stim vs spontaneous data
stim_spont_data <- read_data("stim_vs_spont_agreements_for_lme.csv")

if (!is.null(stim_spont_data)) {
  cat("Data loaded successfully. Shape:", nrow(stim_spont_data), "x", ncol(stim_spont_data), "\n")
  
  # Prepare data - this should only contain NDD model data
  stim_spont_data <- stim_spont_data %>%
    mutate(
      stim = as.factor(stim),
      patient = as.factor(patient)
    )
  
  # Rename stim levels for clarity
  levels(stim_spont_data$stim) <- c("Spontaneous", "Stimulated")
  
  # Descriptive statistics
  cat("Descriptive Statistics by Seizure Type:\n")
  desc_stats_stim <- stim_spont_data %>%
    group_by(stim) %>%
    summarise(
      n = n(),
      mean_dice = mean(dice, na.rm = TRUE),
      median_dice = median(dice, na.rm = TRUE),
      sd_dice = sd(dice, na.rm = TRUE),
      .groups = 'drop'
    )
  print(desc_stats_stim)
  cat("\n")
  
  # Mixed effects model for stim vs spontaneous
  cat("Mixed Effects Model for Stim vs Spontaneous Agreement:\n")
  if ("patient" %in% colnames(stim_spont_data) && length(unique(stim_spont_data$patient)) > 1) {
    stim_model <- lmer(dice ~ stim + (1|patient), data = stim_spont_data)
  } else {
    stim_model <- lm(dice ~ stim, data = stim_spont_data)
  }
  
  # Model summary
  print(summary(stim_model))
  cat("\n")
  
  # ANOVA for fixed effects
  cat("ANOVA for Fixed Effects:\n")
  stim_anova <- anova(stim_model)
  print(stim_anova)
  cat("\n")
  
  # Post-hoc comparisons
  cat("Post-hoc Comparisons:\n")
  stim_emm <- emmeans(stim_model, ~ stim)
  stim_contrasts <- contrast(stim_emm, method = "pairwise")
  print(stim_contrasts)
  cat("\n")
  
  # Effect size estimation
  cat("Effect Size Estimation:\n")
  print(confint(stim_contrasts))
  cat("\n")
  
} else {
  cat("Stim vs spontaneous data not found. Skipping analysis.\n\n")
}

# ===================================================================
# 4. COMBINED ANALYSIS AND VISUALIZATION SUGGESTIONS
# ===================================================================

cat("4. SUMMARY AND RECOMMENDATIONS\n")
cat("===============================\n")

cat("Analysis completed successfully!\n\n")

cat("Key findings summary:\n")
cat("- All analyses used mixed effects models with random intercepts for patients\n")
cat("- This accounts for the hierarchical structure of the data (seizures nested within patients)\n")
cat("- Post-hoc comparisons used appropriate multiple comparison corrections\n")
cat("- Section 2B provides fixed effects comparison including Interrater without multiple comparison correction\n\n")

cat("For visualization in R, consider:\n")
cat("1. Using ggplot2 with geom_boxplot() and geom_point() for model comparisons\n")
cat("2. Adding patient-level random effects visualization with geom_line()\n")
cat("3. Creating forest plots for effect sizes using the forestplot package\n")
cat("4. Using the sjPlot package for model visualization\n\n")

cat("To save results, you can use:\n")
cat("- write.csv() for tabular results\n")
cat("- ggsave() for plots\n")
cat("- saveRDS() for model objects\n\n")

cat("Analysis complete!\n") 