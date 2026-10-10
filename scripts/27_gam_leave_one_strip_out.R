#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
#
# 27_gam_leave_one_strip_out.R
#
# Leave-one-strip-out (LOSO) prediction with the pre-specified GAMs of
# script 26. Companion of the RCS strip-holdout analysis (script 19).
#
# Cross-validation unit
# ---------------------
# One physical strip = one phone x pH cell = five light images.
# Each of the 39 folds removes all five images of one strip, refits every
# model on the remaining 190 images, and predicts the held-out images.
#
# Prediction
# ----------
# The held-out strip is a new random-effect level, so predictions are
# population-level: the strip random effect is set to zero. Light and
# phone:light terms are those of the held-out image.
#
# Models (identical to script 26)
# -------------------------------
# Common: light + sum-coded phone:light + s(strip_id, bs = "re"), REML
#   M0: s(ph, k)
#   M1: M0 + phone                     (comparable to rcs_additive, script 19)
#   M2: M1 + s(ph, by = phone_ord, k)  (comparable to rcs_phone)
#
# Specifications: tp k = 10 (primary); tp k = 6, 8, 12; cr k = 10.
# Regions: contrast (primary), pad, bg.
#
# Each Lab channel is fitted separately; per-image Delta E76 is computed
# from the three channel predictions in 27_summarize_gam_strip_holdout.py.
#
# Outputs
# -------
# outputs/gam_strip_holdout/
#   loso_predictions.csv   one row per held-out image x spec x region x model
#   fold_audit.csv         convergence and fit information per fold
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
output_dir <- resolve(get_arg("output-dir", "outputs/gam_strip_holdout"))
spec_filter <- get_arg("specs", "all")

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


# ---------------------------------------------------------------------------
# Data (same preparation as script 26)
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
if (any(table(data$strip_id) != 5)) {
  stop("Every strip must appear under exactly five lights.")
}

PHONES <- levels(data$phone)
LIGHTS <- levels(data$light)
data$phone_ord <- factor(data$phone, levels = PHONES, ordered = TRUE)

pl_matrix <- function(phone, light) {
  # Sum-coded phone x light product columns, orthogonal to the main effects.
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

# Population-level prediction for new strips: strip random effect = 0.
predict_new_strip <- function(fit, train, test) {
  newdata <- test
  newdata$strip_id <- factor(levels(train$strip_id)[1], levels = levels(train$strip_id))
  X <- predict(fit, newdata = newdata, type = "lpmatrix")
  X[, grep("^s\\(strip_id\\)", colnames(X))] <- 0
  as.vector(X %*% coef(fit))
}


# ---------------------------------------------------------------------------
# LOSO loop
# ---------------------------------------------------------------------------

strips <- unique(data[, c("strip_id", "phone", "ph")])
strips <- strips[order(strips$phone, strips$ph), ]

prediction_rows <- list()
audit_rows <- list()
started <- Sys.time()

for (spec_index in seq_len(nrow(SPECS))) {
  basis <- SPECS$basis[spec_index]
  k <- SPECS$k[spec_index]
  is_primary <- SPECS$primary[spec_index]
  message(sprintf("Spec %s k=%d (%d/%d)", basis, k, spec_index, nrow(SPECS)))

  for (fold in seq_len(nrow(strips))) {
    held_out <- as.character(strips$strip_id[fold])
    test_mask <- as.character(data$strip_id) == held_out
    train <- data[!test_mask, ]
    train$strip_id <- droplevels(train$strip_id)
    test <- data[test_mask, ]

    for (region in REGIONS) {
      for (model_name in MODELS) {
        predicted <- matrix(NA_real_, nrow(test), length(CHANNELS))
        colnames(predicted) <- CHANNELS

        for (channel in CHANNELS) {
          response <- paste0(region, "_", channel)
          fit <- gam(
            model_formula(response, model_name, basis, k),
            data = train,
            method = "REML"
          )
          predicted[, channel] <- predict_new_strip(fit, train, test)

          audit_rows[[length(audit_rows) + 1]] <- data.frame(
            basis = basis, k = k, primary_spec = is_primary,
            fold = fold, strip_id = held_out, region = region,
            channel = channel, model = model_name,
            converged = isTRUE(fit$converged),
            edf_total = sum(fit$edf),
            n_train = nrow(train)
          )
        }

        prediction_rows[[length(prediction_rows) + 1]] <- data.frame(
          basis = basis, k = k, primary_spec = is_primary,
          fold = fold, region = region, model = model_name,
          image_id = test$image_id,
          strip_id = as.character(test$strip_id),
          phone = as.character(test$phone),
          light = as.character(test$light),
          ph_nominal = test$ph,
          actual_L = test[[paste0(region, "_L")]],
          actual_a = test[[paste0(region, "_a")]],
          actual_b = test[[paste0(region, "_b")]],
          predicted_L = predicted[, "L"],
          predicted_a = predicted[, "a"],
          predicted_b = predicted[, "b"]
        )
      }
    }
  }
}

predictions <- do.call(rbind, prediction_rows)
audit <- do.call(rbind, audit_rows)

write.csv(predictions, file.path(output_dir, "loso_predictions.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
write.csv(audit, file.path(output_dir, "fold_audit.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

coverage <- table(predictions$image_id[
  predictions$primary_spec & predictions$region == "contrast" & predictions$model == "M2"
])
if (length(coverage) != 195 || any(coverage != 1)) {
  stop("LOSO coverage failed: every image must be predicted exactly once.")
}

elapsed <- as.numeric(difftime(Sys.time(), started, units = "mins"))
notes <- c(
  "GAM leave-one-strip-out (script 27)",
  strrep("=", 80),
  paste("input:", input_path),
  paste("output:", output_dir),
  paste("R:", R.version.string, "| mgcv:", as.character(packageVersion("mgcv"))),
  paste("folds:", nrow(strips), "| specs:", paste0(SPECS$basis, SPECS$k, collapse = ", ")),
  paste("fits:", nrow(audit), "| non-converged:", sum(!audit$converged)),
  sprintf("elapsed: %.1f min", elapsed),
  "",
  "Prediction excludes the strip random effect (new strip).",
  "Summaries, paired bootstrap and RCS comparison: 27_summarize_gam_strip_holdout.py"
)
writeLines(notes, file.path(output_dir, "analysis_audit.txt"), useBytes = TRUE)
cat(notes, sep = "\n")
