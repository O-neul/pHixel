#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
#
# 31_sim_loso_gam.R
#
# Leave-one-strip-out GAM predictions on the simulated data sets written by
# script 30 (loso_datasets.csv). Used by
#   S2 (LOSO part): is there a spurious M2 gain when A = 0?
#   S3: does GAM fail at pH 1/13 when the truth is smooth (A = 1)?
#
# Per data set: 39 folds x {M1, M2} x 3 Lab channels, tp k = 10, REML;
# strip random effect set to zero for the held-out strip (as script 27).
#
# Output
# ------
# outputs/simulation/s3/gam_strip_metrics.csv
#   one row per data set x held-out strip x model (mean Delta E76 over lights)
#

suppressPackageStartupMessages({
  library(mgcv)
  library(parallel)
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

input_path <- resolve(get_arg("input", "outputs/simulation/s2/loso_datasets.csv"))
output_dir <- resolve(get_arg("output-dir", "outputs/simulation/s3"))
cores <- as.integer(get_arg("cores", as.character(max(1, detectCores() - 2))))
max_datasets <- as.integer(get_arg("max-datasets", "0"))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

CHANNELS <- c("L", "a", "b")
K <- 10L
BASIS <- "tp"

all_data <- read.csv(input_path, fileEncoding = "UTF-8", stringsAsFactors = FALSE)
keys <- unique(all_data[, c("A", "dataset")])
if (max_datasets > 0) {
  keys <- do.call(rbind, lapply(split(keys, keys$A), head, max_datasets))
}

PHONES <- sort(unique(all_data$phone))
LIGHTS <- sort(unique(all_data$light))

worker_setup <- function(PHONES, LIGHTS, K, BASIS, CHANNELS) {
  suppressPackageStartupMessages(library(mgcv))

  pl_matrix <<- function(phone, light) {
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

  PL_NAMES <<- colnames(pl_matrix(PHONES[1], LIGHTS[1]))

  model_formula <<- function(model) {
    fixed <- paste(c("light", PL_NAMES), collapse = " + ")
    smooth <- sprintf("s(ph, bs = '%s', k = %d)", BASIS, K)
    terms <- switch(
      model,
      M1 = c("phone", fixed, smooth),
      M2 = c("phone", fixed, smooth,
             sprintf("s(ph, by = phone_ord, bs = '%s', k = %d)", BASIS, K))
    )
    as.formula(paste("y ~", paste(c(terms, "s(strip_id, bs = 're')"), collapse = " + ")))
  }

  run_dataset <<- function(frame) {
    frame$strip_id <- factor(frame$strip_id)
    frame$phone <- factor(frame$phone, levels = PHONES)
    frame$phone_ord <- factor(frame$phone, levels = PHONES, ordered = TRUE)
    frame$light <- factor(frame$light, levels = LIGHTS)
    frame$ph <- frame$ph_nominal
    frame <- cbind(frame, pl_matrix(frame$phone, frame$light))

    strips <- unique(frame[, c("strip_id", "phone", "ph")])
    rows <- list()

    for (fold in seq_len(nrow(strips))) {
      held_out <- as.character(strips$strip_id[fold])
      mask <- as.character(frame$strip_id) == held_out
      train <- frame[!mask, ]
      train$strip_id <- droplevels(train$strip_id)
      test <- frame[mask, ]
      test$strip_id <- factor(levels(train$strip_id)[1], levels = levels(train$strip_id))

      for (model in c("M1", "M2")) {
        predicted <- matrix(NA_real_, nrow(test), 3)
        for (j in seq_along(CHANNELS)) {
          train$y <- train[[paste0("contrast_", CHANNELS[j])]]
          fit <- gam(model_formula(model), data = train, method = "REML")
          X <- predict(fit, newdata = test, type = "lpmatrix")
          X[, grep("^s\\(strip_id\\)", colnames(X))] <- 0
          predicted[, j] <- as.vector(X %*% coef(fit))
        }
        actual <- as.matrix(test[, paste0("contrast_", CHANNELS)])
        delta_e <- sqrt(rowSums((predicted - actual)^2))
        rows[[length(rows) + 1]] <- data.frame(
          A = frame$A[1], dataset = frame$dataset[1], method = "GAM", variant = "tp k=10",
          model = model, strip_id = held_out,
          phone = as.character(strips$phone[fold]), ph_nominal = strips$ph[fold],
          mean_deltaE76 = mean(delta_e)
        )
      }
    }
    do.call(rbind, rows)
  }
  NULL
}

cluster <- makeCluster(cores)
on.exit(stopCluster(cluster), add = TRUE)
clusterCall(cluster, worker_setup, PHONES, LIGHTS, K, BASIS, CHANNELS)

chunks <- lapply(seq_len(nrow(keys)), function(i) {
  all_data[all_data$A == keys$A[i] & all_data$dataset == keys$dataset[i], ]
})

started <- Sys.time()
results <- parLapply(cluster, chunks, function(frame) {
  tryCatch(run_dataset(frame), error = function(e) NULL)
})
ok <- !vapply(results, is.null, logical(1))
metrics <- do.call(rbind, results[ok])

write.csv(metrics, file.path(output_dir, "gam_strip_metrics.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

elapsed <- as.numeric(difftime(Sys.time(), started, units = "mins"))
notes <- c(
  "Simulation LOSO, GAM (script 31)",
  strrep("=", 80),
  paste("data sets:", nrow(keys), "| succeeded:", sum(ok), "| cores:", cores),
  sprintf("elapsed: %.1f min", elapsed)
)
writeLines(notes, file.path(output_dir, "gam_loso_audit.txt"), useBytes = TRUE)
cat(notes, sep = "\n")
