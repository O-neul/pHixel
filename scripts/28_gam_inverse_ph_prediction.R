#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
#
# 28_gam_inverse_ph_prediction.R
#
# Practical meaning of the phone-specific pH structure: inverse prediction
# of pH from the observed Lab colour of held-out strips.
#
# Script 27 measured forward prediction error (Lab, Delta E76). Here the
# question is whether the Delta E gain translates into better pH readings.
#
# Procedure (same 39 leave-one-strip-out folds as script 27)
# ----------------------------------------------------------
# 1. Fit M0/M1/M2 for L*, a*, b* on the 190 training images.
# 2. For the held-out strip's phone and each light, predict the population
#    Lab trajectory over a pH grid (1 to 13, step 0.05; strip effect = 0).
# 3. For every held-out image, the estimated pH is the grid value whose
#    predicted Lab (same phone, same light) is closest in Delta E76.
#
# Calibration scenarios:
#   M0: one calibration curve shared by all phones (phone unknown)
#   M1: shared curve shape + phone-specific offset
#   M2: phone-specific calibration curves
#
# --light-mode known (default): the image's light is known and its curve is used.
# --light-mode unknown: one curve averaged over the five lights is used.
#
# The grid is restricted to the observed range [1, 13], so pH 1 and 13
# estimates cannot overshoot; boundary errors are reported separately.
#
# Background is included as a negative control: it carries almost no pH
# signal, so its inverse prediction should fail.
#
# Outputs
# -------
# outputs/gam_inverse_ph/
#   inverse_predictions.csv   one row per held-out image x spec x region x model
#   analysis_audit.txt
#

suppressPackageStartupMessages({
  library(mgcv)
})

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(name, default) {
  key <- paste0("--", name)
  hit <- which(args == key)
  if (length(hit) == 0) return(default)
  args[hit + 1]
}

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
project_root <- if (length(script_path) == 1) {
  normalizePath(file.path(dirname(script_path), ".."))
} else {
  getwd()
}

resolve <- function(path) {
  if (grepl("^(/|[A-Za-z]:)", path)) path else file.path(project_root, path)
}

input_path <- resolve(get_arg(
  "input", "outputs/factorial_analysis/analysis_factorial_table.csv"
))
output_dir <- resolve(get_arg("output-dir", "outputs/gam_inverse_ph"))
spec_filter <- get_arg("specs", "all")
grid_step <- as.numeric(get_arg("grid-step", "0.05"))
light_mode <- get_arg("light-mode", "known")
if (!light_mode %in% c("known", "unknown")) stop("--light-mode must be known or unknown.")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

REGIONS <- c("contrast", "pad", "bg")
CHANNELS <- c("L", "a", "b")
MODELS <- c("M0", "M1", "M2")
SPECS <- data.frame(
  basis = c("tp", "tp", "tp", "tp", "cr"),
  k = c(10L, 6L, 8L, 12L, 10L),
  stringsAsFactors = FALSE
)
SPECS$primary <- SPECS$basis == "tp" & SPECS$k == 10L
if (spec_filter == "primary") SPECS <- SPECS[SPECS$primary, ]

PH_GRID <- seq(1, 13, by = grid_step)


# ---------------------------------------------------------------------------
# Data and model helpers (same as scripts 26-27)
# ---------------------------------------------------------------------------

raw <- read.csv(input_path, fileEncoding = "UTF-8-BOM", stringsAsFactors = FALSE)

required <- c(
  "image_id", "strip_id", "phone", "light", "ph_nominal",
  as.vector(outer(REGIONS, CHANNELS, function(r, ch) paste0(r, "_", ch)))
)
missing <- setdiff(required, names(raw))
if (length(missing) > 0) stop("Missing columns: ", paste(missing, collapse = ", "))

data <- data.frame(
  image_id = as.character(raw$image_id),
  strip_id = factor(raw$strip_id),
  phone = factor(raw$phone),
  light = factor(raw$light),
  ph = as.numeric(raw$ph_nominal),
  stringsAsFactors = FALSE
)
for (region in REGIONS) {
  for (channel in CHANNELS) {
    column <- paste0(region, "_", channel)
    data[[column]] <- as.numeric(raw[[column]])
  }
}

cells <- table(data$ph, data$phone, data$light)
if (nrow(data) != 195 || any(cells != 1)) {
  stop("Expected a complete 13 x 3 x 5 design with one image per cell.")
}

PHONES <- levels(data$phone)
LIGHTS <- levels(data$light)
data$phone_ord <- factor(data$phone, levels = PHONES, ordered = TRUE)

pl_matrix <- function(phone, light) {
  phone_index <- as.integer(factor(phone, levels = PHONES))
  light_index <- as.integer(factor(light, levels = LIGHTS))
  cp <- contr.sum(length(PHONES))[phone_index, , drop = FALSE]
  cl <- contr.sum(length(LIGHTS))[light_index, , drop = FALSE]
  out <- do.call(cbind, lapply(seq_len(ncol(cp)), function(i) cp[, i] * cl))
  colnames(out) <- paste0(
    "pl", rep(seq_len(ncol(cp)), each = ncol(cl)), "_", seq_len(ncol(cl))
  )
  out
}

PL <- pl_matrix(data$phone, data$light)
PL_NAMES <- colnames(PL)
data <- cbind(data, PL)

model_formula <- function(response, model, basis, k) {
  fixed <- paste(c("light", PL_NAMES), collapse = " + ")
  smooth <- sprintf("s(ph, bs = '%s', k = %d)", basis, k)
  re <- "s(strip_id, bs = 're')"

  terms <- switch(
    model,
    M0 = c(fixed, smooth, re),
    M1 = c("phone", fixed, smooth, re),
    M2 = c(
      "phone", fixed, smooth,
      sprintf("s(ph, by = phone_ord, bs = '%s', k = %d)", basis, k),
      re
    )
  )

  as.formula(paste(response, "~", paste(terms, collapse = " + ")))
}

# Calibration grid for one phone: every light x every pH grid value.
calibration_grid <- function(train, phone_value) {
  grid <- expand.grid(
    ph = PH_GRID,
    light = factor(LIGHTS, levels = LIGHTS),
    KEEP.OUT.ATTRS = FALSE
  )
  grid$phone <- factor(phone_value, levels = PHONES)
  grid$phone_ord <- factor(phone_value, levels = PHONES, ordered = TRUE)
  grid$strip_id <- factor(levels(train$strip_id)[1], levels = levels(train$strip_id))
  cbind(grid, pl_matrix(grid$phone, grid$light))
}

predict_population <- function(fit, newdata) {
  X <- predict(fit, newdata = newdata, type = "lpmatrix")
  X[, grep("^s\\(strip_id\\)", colnames(X))] <- 0
  as.vector(X %*% coef(fit))
}


# ---------------------------------------------------------------------------
# LOSO inverse prediction
# ---------------------------------------------------------------------------

strips <- unique(data[, c("strip_id", "phone", "ph")])
strips <- strips[order(strips$phone, strips$ph), ]

rows <- list()
n_fits <- 0L
n_nonconverged <- 0L
started <- Sys.time()

for (spec_index in seq_len(nrow(SPECS))) {
  basis <- SPECS$basis[spec_index]
  k <- SPECS$k[spec_index]
  is_primary <- SPECS$primary[spec_index]
  message(sprintf("Spec %s k=%d (%d/%d)", basis, k, spec_index, nrow(SPECS)))

  for (fold in seq_len(nrow(strips))) {
    held_out <- as.character(strips$strip_id[fold])
    phone_value <- as.character(strips$phone[fold])
    test_mask <- as.character(data$strip_id) == held_out
    train <- data[!test_mask, ]
    train$strip_id <- droplevels(train$strip_id)
    test <- data[test_mask, ]
    grid <- calibration_grid(train, phone_value)

    for (region in REGIONS) {
      for (model_name in MODELS) {
        curve <- matrix(NA_real_, nrow(grid), length(CHANNELS))
        colnames(curve) <- CHANNELS

        for (channel in CHANNELS) {
          fit <- gam(
            model_formula(paste0(region, "_", channel), model_name, basis, k),
            data = train,
            method = "REML"
          )
          n_fits <- n_fits + 1L
          if (!isTRUE(fit$converged)) n_nonconverged <- n_nonconverged + 1L
          curve[, channel] <- predict_population(fit, grid)
        }

        # Light unknown: one calibration curve averaged over the five lights.
        # The grid is ordered with pH varying fastest within each light.
        curve_average <- sapply(CHANNELS, function(channel) {
          rowMeans(matrix(curve[, channel], nrow = length(PH_GRID)))
        })

        for (i in seq_len(nrow(test))) {
          if (light_mode == "known") {
            reference <- curve[grid$light == as.character(test$light[i]), , drop = FALSE]
          } else {
            reference <- curve_average
          }
          observed <- c(
            test[[paste0(region, "_L")]][i],
            test[[paste0(region, "_a")]][i],
            test[[paste0(region, "_b")]][i]
          )
          distance <- sqrt(colSums((t(reference) - observed)^2))
          best <- which.min(distance)

          rows[[length(rows) + 1]] <- data.frame(
            basis = basis, k = k, primary_spec = is_primary,
            fold = fold, region = region, model = model_name,
            image_id = test$image_id[i],
            strip_id = held_out,
            phone = phone_value,
            light = as.character(test$light[i]),
            true_ph = test$ph[i],
            light_mode = light_mode,
            estimated_ph = PH_GRID[best],
            min_deltaE76 = distance[best]
          )
        }
      }
    }
  }
}

result <- do.call(rbind, rows)
result$error_ph <- result$estimated_ph - result$true_ph
result$abs_error_ph <- abs(result$error_ph)

write.csv(result, file.path(output_dir, "inverse_predictions.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

coverage <- table(result$image_id[
  result$primary_spec & result$region == "contrast" & result$model == "M2"
])
if (length(coverage) != 195 || any(coverage != 1)) {
  stop("Coverage failed: every image must be predicted exactly once.")
}

elapsed <- as.numeric(difftime(Sys.time(), started, units = "mins"))
notes <- c(
  "GAM inverse pH prediction (script 28)",
  strrep("=", 80),
  paste("input:", input_path),
  paste("output:", output_dir),
  paste("R:", R.version.string, "| mgcv:", as.character(packageVersion("mgcv"))),
  paste("folds:", nrow(strips), "| specs:", paste0(SPECS$basis, SPECS$k, collapse = ", ")),
  paste("pH grid:", min(PH_GRID), "to", max(PH_GRID), "step", grid_step),
  paste("light mode:", light_mode),
  paste("fits:", n_fits, "| non-converged:", n_nonconverged),
  sprintf("elapsed: %.1f min", elapsed),
  "",
  "Estimated pH = grid value minimizing Delta E76 to the predicted Lab of the",
  "held-out strip's phone under the image's light (strip effect = 0).",
  "Summaries: 28_summarize_inverse_ph.py"
)
writeLines(notes, file.path(output_dir, "analysis_audit.txt"), useBytes = TRUE)
cat(notes, sep = "\n")
