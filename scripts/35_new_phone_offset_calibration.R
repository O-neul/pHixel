#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
#
# 35_new_phone_offset_calibration.R
#
# How many reference strips does a new phone need for an offset calibration?
#
# Leave-one-phone-out:
#   1. Fit the shared-curve GAM (M1 structure, tp k = 10, strip random effect)
#      on the two other phones. The base curve for the new phone is the
#      average of the two training phones (phone offset unknown).
#   2. Photograph k reference strips of known pH with the new phone
#      (each strip under its five lights). The phone offset is the mean Lab
#      residual of those images from the base curve.
#   3. Read the pH of the remaining strips of the new phone from
#      base curve + offset (nearest point on a pH grid, Delta E76).
#
# k = 0 (no calibration), 1, 2, 3 (every combination of reference pH),
# and k = 12 (all other strips of the phone; leave-one-strip-out).
# Light known (curve of the image's light) and light unknown (curve averaged
# over lights) are both reported.
#
# Only the offset is calibrated: a phone-specific curve shape cannot be
# estimated from a few strips.
#
# Outputs
# -------
# outputs/new_phone_calibration/
#   calibration_results.csv   one row per target phone x k x reference set x light mode
#   image_predictions.csv     k = 0 and k = 12 image-level predictions
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
output_dir <- resolve(get_arg("output-dir", "outputs/new_phone_calibration"))
region <- get_arg("region", "contrast")
max_k <- as.integer(get_arg("max-k", "3"))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

CHANNELS <- c("L", "a", "b")
K <- 10L
BASIS <- "tp"
PH_GRID <- seq(1, 13, by = 0.05)

raw <- read.csv(input_path, fileEncoding = "UTF-8-BOM", stringsAsFactors = FALSE)
data <- data.frame(
  image_id = as.character(raw$image_id),
  strip_id = as.character(raw$strip_id),
  phone = as.character(raw$phone),
  light = as.character(raw$light),
  ph = as.numeric(raw$ph_nominal),
  stringsAsFactors = FALSE
)
for (channel in CHANNELS) {
  data[[channel]] <- as.numeric(raw[[paste0(region, "_", channel)]])
}

PHONES <- sort(unique(data$phone))
LIGHTS <- sort(unique(data$light))
PH_LEVELS <- sort(unique(data$ph))


# ---------------------------------------------------------------------------
# Base curve from the two training phones
# ---------------------------------------------------------------------------

base_curves <- function(train) {
  train$strip_id <- factor(train$strip_id)
  train$phone <- factor(train$phone)
  train$light <- factor(train$light, levels = LIGHTS)
  options(contrasts = c("contr.sum", "contr.poly"))
  on.exit(options(contrasts = c("contr.treatment", "contr.poly")), add = TRUE)

  curve <- array(NA_real_, c(length(PH_GRID), length(LIGHTS), 3))
  for (j in seq_along(CHANNELS)) {
    train$y <- train[[CHANNELS[j]]]
    fit <- gam(
      y ~ phone * light + s(ph, bs = BASIS, k = K) + s(strip_id, bs = "re"),
      data = train, method = "REML"
    )
    for (l in seq_along(LIGHTS)) {
      per_phone <- sapply(levels(train$phone), function(p) {
        newdata <- data.frame(
          ph = PH_GRID,
          phone = factor(p, levels = levels(train$phone)),
          light = factor(LIGHTS[l], levels = LIGHTS),
          strip_id = factor(levels(train$strip_id)[1], levels = levels(train$strip_id))
        )
        X <- predict(fit, newdata = newdata, type = "lpmatrix")
        X[, grep("^s\\(strip_id\\)", colnames(X))] <- 0
        as.vector(X %*% coef(fit))
      })
      curve[, l, j] <- rowMeans(per_phone)   # phone-average: new phone offset unknown
    }
  }
  curve
}

# Base value at the exact pH of an image (grid contains every integer pH).
grid_index <- function(ph) match(round(ph * 20), round(PH_GRID * 20))

read_ph <- function(observed, light, curve, offset, light_mode) {
  # observed: n x 3; returns estimated pH per row
  reference_all <- curve + array(rep(offset, each = length(PH_GRID) * length(LIGHTS)),
                                 dim(curve))
  averaged <- apply(reference_all, c(1, 3), mean)            # grid x 3
  sapply(seq_len(nrow(observed)), function(i) {
    reference <- if (light_mode == "known") {
      reference_all[, match(light[i], LIGHTS), ]
    } else {
      averaged
    }
    distance <- sqrt(rowSums((reference - matrix(observed[i, ], nrow(reference), 3,
                                                  byrow = TRUE))^2))
    PH_GRID[which.min(distance)]
  })
}

offset_from <- function(reference, curve) {
  if (nrow(reference) == 0) return(c(0, 0, 0))
  base <- t(sapply(seq_len(nrow(reference)), function(i) {
    curve[grid_index(reference$ph[i]), match(reference$light[i], LIGHTS), ]
  }))
  colMeans(as.matrix(reference[, CHANNELS]) - base)
}

metrics <- function(test, estimated) {
  error <- estimated - test$ph
  interior <- test$ph >= 2 & test$ph <= 12
  strip_error <- tapply(abs(error), test$strip_id, mean)
  strip_ph <- tapply(test$ph, test$strip_id, mean)
  c(
    mean_abs_error_interior = mean(strip_error[strip_ph >= 2 & strip_ph <= 12]),
    mean_abs_error_all = mean(strip_error),
    exact_accuracy_interior = mean((floor(estimated + 0.5) == test$ph)[interior]),
    within1_accuracy_interior = mean((abs(error) <= 1 + 1e-9)[interior]),
    mean_bias_interior = mean(error[interior])
  )
}


# ---------------------------------------------------------------------------
# Leave-one-phone-out
# ---------------------------------------------------------------------------

result_rows <- list()
image_rows <- list()
started <- Sys.time()

for (target in PHONES) {
  message("Target phone: ", target)
  curve <- base_curves(data[data$phone != target, ])
  phone_data <- data[data$phone == target, ]

  for (light_mode in c("known", "unknown")) {
    # k = 0: no calibration
    estimated <- read_ph(as.matrix(phone_data[, CHANNELS]), phone_data$light, curve,
                         c(0, 0, 0), light_mode)
    result_rows[[length(result_rows) + 1]] <- data.frame(
      region = region, target_phone = target, light_mode = light_mode, k = 0L,
      reference_ph = "", t(metrics(phone_data, estimated))
    )
    image_rows[[length(image_rows) + 1]] <- data.frame(
      region = region, target_phone = target, light_mode = light_mode, k = 0L,
      image_id = phone_data$image_id, true_ph = phone_data$ph, estimated_ph = estimated
    )

    # k = 1..max_k: every combination of reference pH levels
    for (k in seq_len(max_k)) {
      combos <- combn(PH_LEVELS, k)
      for (column in seq_len(ncol(combos))) {
        reference_ph <- combos[, column]
        reference <- phone_data[phone_data$ph %in% reference_ph, ]
        test <- phone_data[!phone_data$ph %in% reference_ph, ]
        offset <- offset_from(reference, curve)
        estimated <- read_ph(as.matrix(test[, CHANNELS]), test$light, curve, offset,
                             light_mode)
        result_rows[[length(result_rows) + 1]] <- data.frame(
          region = region, target_phone = target, light_mode = light_mode, k = k,
          reference_ph = paste(reference_ph, collapse = ","), t(metrics(test, estimated))
        )
      }
    }

    # k = 12: all other strips of the phone (leave-one-strip-out)
    estimated <- numeric(nrow(phone_data))
    for (ph in PH_LEVELS) {
      held <- phone_data$ph == ph
      offset <- offset_from(phone_data[!held, ], curve)
      estimated[held] <- read_ph(as.matrix(phone_data[held, CHANNELS]),
                                 phone_data$light[held], curve, offset, light_mode)
    }
    result_rows[[length(result_rows) + 1]] <- data.frame(
      region = region, target_phone = target, light_mode = light_mode,
      k = length(PH_LEVELS) - 1L, reference_ph = "all other strips",
      t(metrics(phone_data, estimated))
    )
    image_rows[[length(image_rows) + 1]] <- data.frame(
      region = region, target_phone = target, light_mode = light_mode,
      k = length(PH_LEVELS) - 1L,
      image_id = phone_data$image_id, true_ph = phone_data$ph, estimated_ph = estimated
    )
  }
}

results <- do.call(rbind, result_rows)
write.csv(results, file.path(output_dir, paste0("calibration_results_", region, ".csv")),
          row.names = FALSE, fileEncoding = "UTF-8")
write.csv(do.call(rbind, image_rows),
          file.path(output_dir, paste0("image_predictions_", region, ".csv")),
          row.names = FALSE, fileEncoding = "UTF-8")

summary <- aggregate(
  cbind(mean_abs_error_interior, exact_accuracy_interior) ~ light_mode + k,
  data = results, FUN = function(x) c(mean = mean(x), q10 = unname(quantile(x, 0.1)),
                                      q90 = unname(quantile(x, 0.9)))
)
elapsed <- as.numeric(difftime(Sys.time(), started, units = "mins"))
notes <- c(
  "New-phone offset calibration (script 35)",
  strrep("=", 80),
  paste("region:", region, "| base model: M1 structure, tp k = 10, REML"),
  paste("combinations evaluated:", nrow(results)),
  sprintf("elapsed: %.1f min", elapsed),
  "",
  "pH 2-12 mean |pH error| and exact accuracy over target phones and reference sets",
  "(mean, 10th and 90th percentile across reference sets):",
  capture.output(print(summary, digits = 3))
)
writeLines(notes, file.path(output_dir, paste0("analysis_audit_", region, ".txt")),
           useBytes = TRUE)
cat(notes, sep = "\n")
