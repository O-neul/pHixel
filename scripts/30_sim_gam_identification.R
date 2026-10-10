#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
#
# 30_sim_gam_identification.R
#
# Simulation S2 (docs/SIMULATION_DESIGN.md): can the script 26 GAM separate
# smooth phone-specific pH curves from non-smooth strip deviations in the
# 13 pH x 3 phone x 5 light design with one strip per phone x pH?
#
# Truth (per channel, calibrated to the real Contrast fit, tp k = 10)
# ------------------------------------------------------------------
#   mu_A = population M2 fit - (1 - A) * phone-specific curve terms
#   y    = mu_A + u_strip ~ N(0, (m * sigma_s)^2) + e ~ N(0, sigma_e^2)
#
# A scales the phone-specific curve shapes (A = 1: real size, A = 0: none).
# m scales the real non-smooth strip SD (sigma_s from M2).
#
# Cells
#   a*:     A in {0, 0.25, 0.5, 1} x m in {0.5, 1, 2}
#   L*, b*: A in {0, 1} x m = 1
#   500 data sets per cell.
#
# Fitted models (as script 26)
#   M1, M2 with strip random effect (REML)
#   M2_naive: M2 without the strip random effect
#
# Recorded per data set
#   - approximate p-values of the two phone-specific curve terms (M2, M2_naive)
#   - strip variance reduction 1 - sigma_s^2(M2) / sigma_s^2(M1)
#   - RMSE of each estimated phone-specific curve vs truth on a pH grid
#   - pointwise and simultaneous 95% band coverage of each curve
#
# Replicates run on socket workers (parallel::parLapply).
#
# The script also writes three-channel data sets for the LOSO simulations
# (scripts 31): A in {0, 1}, m = 1, 200 data sets each.
#
# Outputs
# -------
# outputs/simulation/s2/
#   s2_replicates.csv
#   s2_truth_parameters.csv
#   loso_datasets.csv
#   s2_audit.txt
#

suppressPackageStartupMessages({
  library(mgcv)
  library(MASS)
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

input_path <- resolve(get_arg(
  "input", "outputs/factorial_analysis/analysis_factorial_table.csv"
))
output_dir <- resolve(get_arg("output-dir", "outputs/simulation/s2"))
n_reps <- as.integer(get_arg("n-reps", "500"))
n_loso <- as.integer(get_arg("n-loso-datasets", "200"))
n_sim <- as.integer(get_arg("n-sim", "1000"))
cores <- as.integer(get_arg("cores", as.character(max(1, detectCores() - 1))))
seed <- as.integer(get_arg("seed", "20261009"))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

CHANNELS <- c("L", "a", "b")
K <- 10L
BASIS <- "tp"
PH_GRID <- seq(1, 13, by = 0.1)


# ---------------------------------------------------------------------------
# Real data and model helpers (same as scripts 26-28)
# ---------------------------------------------------------------------------

raw <- read.csv(input_path, fileEncoding = "UTF-8-BOM", stringsAsFactors = FALSE)

data <- data.frame(
  image_id = as.character(raw$image_id),
  strip_id = factor(raw$strip_id),
  phone = factor(raw$phone),
  light = factor(raw$light),
  ph = as.numeric(raw$ph_nominal),
  stringsAsFactors = FALSE
)
for (channel in CHANNELS) {
  data[[paste0("contrast_", channel)]] <- as.numeric(raw[[paste0("contrast_", channel)]])
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

model_formula <- function(model, random_effect = TRUE) {
  fixed <- paste(c("light", PL_NAMES), collapse = " + ")
  smooth <- sprintf("s(ph, bs = '%s', k = %d)", BASIS, K)
  terms <- switch(
    model,
    M1 = c("phone", fixed, smooth),
    M2 = c("phone", fixed, smooth,
           sprintf("s(ph, by = phone_ord, bs = '%s', k = %d)", BASIS, K))
  )
  if (random_effect) terms <- c(terms, "s(strip_id, bs = 're')")
  as.formula(paste("y ~", paste(terms, collapse = " + ")))
}

strip_sd <- function(fit) {
  invisible(capture.output(vc <- gam.vcomp(fit)))
  sd <- if (is.list(vc)) vc$all else vc[, "std.dev"]
  unname(sd[grep("strip_id", names(sd))])
}

BY_TERMS <- paste0("s(ph):phone_ord", PHONES[-1])

grid_frame <- function(phone_value) {
  grid <- data.frame(
    ph = PH_GRID,
    phone = factor(phone_value, levels = PHONES),
    phone_ord = factor(phone_value, levels = PHONES, ordered = TRUE),
    light = factor(LIGHTS[1], levels = LIGHTS),
    strip_id = factor(levels(data$strip_id)[1], levels = levels(data$strip_id))
  )
  cbind(grid, pl_matrix(grid$phone, grid$light))
}


# ---------------------------------------------------------------------------
# Truth from the real Contrast fit
# ---------------------------------------------------------------------------

truth <- list()
truth_rows <- list()

for (channel in CHANNELS) {
  fit_data <- data
  fit_data$y <- data[[paste0("contrast_", channel)]]
  real <- gam(model_formula("M2"), data = fit_data, method = "REML")

  population <- as.vector(predict(real, type = "link", exclude = "s(strip_id)"))
  term_matrix <- predict(real, type = "terms")
  curve_terms <- rowSums(term_matrix[, BY_TERMS, drop = FALSE])

  grid_truth <- lapply(PHONES[-1], function(p) {
    terms <- predict(real, newdata = grid_frame(p), type = "terms")
    as.vector(terms[, paste0("s(ph):phone_ord", p)])
  })
  names(grid_truth) <- PHONES[-1]

  truth[[channel]] <- list(
    population = population,
    curve_terms = curve_terms,
    grid_truth = grid_truth,
    sigma_e = sqrt(real$sig2),
    sigma_s = strip_sd(real)
  )

  truth_rows[[channel]] <- data.frame(
    channel = channel,
    sigma_e = sqrt(real$sig2),
    sigma_s_M2 = strip_sd(real),
    curve_term_sd = sd(curve_terms),
    curve_range_iphone13pro = diff(range(grid_truth[["iphone13pro"]])),
    curve_range_iphone16pro = diff(range(grid_truth[["iphone16pro"]]))
  )
}

write.csv(do.call(rbind, truth_rows), file.path(output_dir, "s2_truth_parameters.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

strip_index <- as.integer(data$strip_id)
n_strips <- nlevels(data$strip_id)

simulate_y <- function(channel, A, m) {
  tr <- truth[[channel]]
  mu <- tr$population - (1 - A) * tr$curve_terms
  u <- rnorm(n_strips, sd = m * tr$sigma_s)
  mu + u[strip_index] + rnorm(nrow(data), sd = tr$sigma_e)
}


# ---------------------------------------------------------------------------
# One replicate
# ---------------------------------------------------------------------------

grids <- lapply(PHONES[-1], grid_frame)
names(grids) <- PHONES[-1]

curve_metrics <- function(fit, channel, A) {
  out <- list()
  beta <- coef(fit)
  Vp <- vcov(fit, unconditional = TRUE)
  draws <- mvrnorm(n_sim, beta, Vp)

  for (p in PHONES[-1]) {
    X <- predict(fit, newdata = grids[[p]], type = "lpmatrix")
    columns <- grep(paste0("^s\\(ph\\):phone_ord", p, "\\."), colnames(X))
    Xs <- X[, columns, drop = FALSE]
    estimate <- as.vector(Xs %*% beta[columns])
    se <- sqrt(pmax(rowSums((Xs %*% Vp[columns, columns]) * Xs), 0))
    deviations <- Xs %*% t(draws[, columns, drop = FALSE] -
                             matrix(beta[columns], n_sim, length(columns), byrow = TRUE))
    crit <- as.numeric(quantile(apply(abs(deviations / pmax(se, 1e-12)), 2, max), 0.95))
    target <- A * truth[[channel]]$grid_truth[[p]]

    out[[p]] <- c(
      rmse = sqrt(mean((estimate - target)^2)),
      pointwise_coverage = mean(abs(estimate - target) <= 1.96 * se),
      simultaneous_coverage = as.numeric(all(abs(estimate - target) <= crit * se))
    )
  }
  out
}

by_p_values <- function(fit) {
  table <- summary(fit)$s.table
  sapply(BY_TERMS, function(term) table[term, "p-value"])
}

run_replicate <- function(channel, A, m, rep_seed) {
  set.seed(rep_seed)
  sim <- data
  sim$y <- simulate_y(channel, A, m)

  m1 <- gam(model_formula("M1"), data = sim, method = "REML")
  m2 <- gam(model_formula("M2"), data = sim, method = "REML")
  naive <- gam(model_formula("M2", random_effect = FALSE), data = sim, method = "REML")

  p_m2 <- by_p_values(m2)
  p_naive <- by_p_values(naive)
  curves <- curve_metrics(m2, channel, A)

  data.frame(
    channel = channel, A = A, strip_multiplier = m, seed = rep_seed,
    p_m2_iphone13pro = p_m2[1], p_m2_iphone16pro = p_m2[2],
    p_naive_iphone13pro = p_naive[1], p_naive_iphone16pro = p_naive[2],
    strip_var_M1 = strip_sd(m1)^2, strip_var_M2 = strip_sd(m2)^2,
    rmse_iphone13pro = curves[["iphone13pro"]][["rmse"]],
    rmse_iphone16pro = curves[["iphone16pro"]][["rmse"]],
    pointwise_cov_iphone13pro = curves[["iphone13pro"]][["pointwise_coverage"]],
    pointwise_cov_iphone16pro = curves[["iphone16pro"]][["pointwise_coverage"]],
    simultaneous_cov_iphone13pro = curves[["iphone13pro"]][["simultaneous_coverage"]],
    simultaneous_cov_iphone16pro = curves[["iphone16pro"]][["simultaneous_coverage"]],
    converged = isTRUE(m1$converged) && isTRUE(m2$converged) && isTRUE(naive$converged),
    row.names = NULL
  )
}


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

cells <- rbind(
  expand.grid(channel = "a", A = c(0, 0.25, 0.5, 1), m = c(0.5, 1, 2),
              stringsAsFactors = FALSE),
  expand.grid(channel = c("L", "b"), A = c(0, 1), m = 1, stringsAsFactors = FALSE)
)

started <- Sys.time()
results <- list()

# Socket workers: forked workers (mclapply) crash with macOS Accelerate BLAS.
cluster <- makeCluster(cores)
on.exit(stopCluster(cluster), add = TRUE)
clusterEvalQ(cluster, {
  suppressPackageStartupMessages({ library(mgcv); library(MASS) })
  NULL
})
clusterExport(cluster, ls(globalenv()), envir = globalenv())

for (cell in seq_len(nrow(cells))) {
  channel <- cells$channel[cell]
  A <- cells$A[cell]
  m <- cells$m[cell]
  seeds <- seed + cell * 100000L + seq_len(n_reps)

  rows <- parLapply(cluster, seeds, function(s, channel, A, m) {
    tryCatch(run_replicate(channel, A, m, s), error = function(e) NULL)
  }, channel = channel, A = A, m = m)
  rows <- Filter(Negate(is.null), rows)
  results[[cell]] <- do.call(rbind, rows)

  message(sprintf("cell %d/%d channel=%s A=%.2f m=%.1f: %d/%d ok (%.1f min)",
                  cell, nrow(cells), channel, A, m, length(rows), n_reps,
                  as.numeric(difftime(Sys.time(), started, units = "mins"))))
}

replicates <- do.call(rbind, results)
write.csv(replicates, file.path(output_dir, "s2_replicates.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")


# ---------------------------------------------------------------------------
# Three-channel data sets for the LOSO simulations (scripts 31)
# ---------------------------------------------------------------------------

loso_rows <- list()
for (A in c(0, 1)) {
  for (r in seq_len(n_loso)) {
    set.seed(seed + 900000L + as.integer(A) * 10000L + r)
    frame <- data.frame(
      A = A, dataset = r,
      image_id = data$image_id, strip_id = as.character(data$strip_id),
      phone = as.character(data$phone), light = as.character(data$light),
      ph_nominal = data$ph
    )
    for (channel in CHANNELS) {
      frame[[paste0("contrast_", channel)]] <- simulate_y(channel, A, 1)
    }
    loso_rows[[length(loso_rows) + 1]] <- frame
  }
}
write.csv(do.call(rbind, loso_rows), file.path(output_dir, "loso_datasets.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

elapsed <- as.numeric(difftime(Sys.time(), started, units = "mins"))
notes <- c(
  "Simulation S2: GAM identification of phone-specific curves (script 30)",
  strrep("=", 80),
  paste("R:", R.version.string, "| mgcv:", as.character(packageVersion("mgcv"))),
  paste("replicates per cell:", n_reps, "| posterior draws:", n_sim, "| cores:", cores),
  paste("rows written:", nrow(replicates), "| non-converged:", sum(!replicates$converged)),
  paste("LOSO data sets:", 2 * n_loso, "(A = 0 and A = 1, m = 1)"),
  sprintf("elapsed: %.1f min", elapsed),
  "Summaries: 32_summarize_simulation.py"
)
writeLines(notes, file.path(output_dir, "s2_audit.txt"), useBytes = TRUE)
cat(notes, sep = "\n")
