#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
#
# 26_gam_phone_specific_curves.R
#
# Phone-specific pH response curves with generalized additive mixed models.
#
# Why GAM after ASCA
# ------------------
# With one strip per phone x pH cell, ASCA has zero strip-level df for
# pH x phone, so phone-specific response cannot be separated from
# strip-to-strip variation (script 25). Assuming the phone-specific response
# is smooth in pH, the GAM splits the pH x phone deviation into
#   - a smooth phone-specific curve, and
#   - non-smooth strip deviations (strip random effect),
# which provides a strip-level error term.
#
# Pre-specified models (docs/NEXT_STEPS_GAM_PLAN.md, fixed before fitting)
# -------------------------------------------------------------------------
# Common: light + phone:light (sum-coded, orthogonal to main effects)
#         + s(strip_id, bs = "re"), method = "REML"
#
#   M0: s(ph, k)                                   no phone term
#   M1: M0 + phone                                 constant phone offset
#   M2: M1 + s(ph, by = phone_ord, k)              phone-specific curves
#
# phone_ord is an ordered factor, so the by-smooths are difference curves
# relative to the reference phone (first level).
#
# Primary specification: basis = "tp", k = 10, region = contrast.
# Sensitivity: k = 6, 8, 12 (tp) and k = 10 (cr); regions pad and bg.
#
# Curves and difference curves
# ----------------------------
# Population-level curves exclude the strip random effect and average the
# light and phone:light terms over the five lights.
# Intervals: pointwise 95% and simultaneous 95% (posterior simulation,
# Marra & Wood 2012 / Wood 2017 section 6.10).
#
# Outputs
# -------
# outputs/gam_curves/
#   model_comparison.csv       AIC (ML refit), REML, deviance explained, r2
#   smooth_terms.csv           edf, Ref.df, F, approximate p for each smooth
#   variance_components.csv    strip and residual SD; strip variance reduction
#   phone_curves.csv           phone-specific curves on a pH grid
#   difference_curves.csv      pairwise phone difference curves
#                              component = total (offset + pH-dependent)
#                              component = shape (pH-dependent part only)
#   difference_regions.csv     pH intervals where the simultaneous CI excludes 0
#   k_check.csv                basis dimension checks
#   concurvity.csv             concurvity of smooth terms
#   fitted_residuals.csv       primary-spec M2 fitted values and residuals
#   strip_effects.csv          strip random-effect predictions (M1, M2)
#   analysis_audit.txt
#

suppressPackageStartupMessages({
  library(mgcv)
  library(MASS)
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
output_dir <- resolve(get_arg("output-dir", "outputs/gam_curves"))
n_sim <- as.integer(get_arg("n-sim", "10000"))
seed <- as.integer(get_arg("seed", "20261009"))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

REGIONS <- c("contrast", "pad", "bg")
CHANNELS <- c("L", "a", "b")
MODELS <- c("M0", "M1", "M2")
PRIMARY <- list(basis = "tp", k = 10L)
SPECS <- data.frame(
  basis = c("tp", "tp", "tp", "tp", "cr"),
  k = c(10L, 6L, 8L, 12L, 10L),
  stringsAsFactors = FALSE
)
SPECS$primary <- SPECS$basis == PRIMARY$basis & SPECS$k == PRIMARY$k
PH_GRID <- seq(1, 13, by = 0.1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

raw <- read.csv(input_path, fileEncoding = "UTF-8-BOM", stringsAsFactors = FALSE)

required <- c(
  "strip_id", "phone", "light", "ph_nominal",
  as.vector(outer(REGIONS, CHANNELS, function(r, ch) paste0(r, "_", ch)))
)
missing <- setdiff(required, names(raw))
if (length(missing) > 0) stop("Missing columns: ", paste(missing, collapse = ", "))

data <- data.frame(
  strip_id = factor(raw$strip_id),
  phone = factor(raw$phone),
  light = factor(raw$light),
  ph = as.numeric(raw$ph_nominal)
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

# Sum-coded phone x light interaction columns. In the balanced design they
# are orthogonal to the phone and light main effects, so M0 contains no
# implicit phone main effect.
pl_matrix <- function(phone, light) {
  # model.matrix(~ phone:light) would expand the interaction fully (and so
  # absorb the phone main effect); build the sum-coded product columns.
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

stopifnot(max(abs(crossprod(PL, model.matrix(~ phone + light, data)[, -1]))) < 1e-10)


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

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

fit_model <- function(response, model, basis, k, method = "REML") {
  gam(model_formula(response, model, basis, k), data = data, method = method)
}


# ---------------------------------------------------------------------------
# Population-level prediction
# ---------------------------------------------------------------------------

# Average linear-predictor matrix over lights, strip random effect excluded.
population_lp <- function(model, phone_value, ph_values) {
  re_columns <- grep("^s\\(strip_id\\)", names(coef(model)))
  accumulated <- NULL

  for (light_value in LIGHTS) {
    newdata <- data.frame(
      ph = ph_values,
      phone = factor(phone_value, levels = PHONES),
      phone_ord = factor(phone_value, levels = PHONES, ordered = TRUE),
      light = factor(light_value, levels = LIGHTS),
      strip_id = factor(levels(data$strip_id)[1], levels = levels(data$strip_id))
    )
    newdata <- cbind(
      newdata,
      pl_matrix(rep(phone_value, length(ph_values)), rep(light_value, length(ph_values)))
    )
    X <- predict(model, newdata = newdata, type = "lpmatrix")
    X[, re_columns] <- 0
    accumulated <- if (is.null(accumulated)) X else accumulated + X
  }

  accumulated / length(LIGHTS)
}

interval_frame <- function(X, model, draws) {
  beta <- coef(model)
  Vp <- vcov(model, unconditional = TRUE)
  fit <- as.vector(X %*% beta)
  se <- sqrt(pmax(rowSums((X %*% Vp) * X), 0))

  # Simultaneous interval: max standardized deviation over the grid.
  deviations <- X %*% t(draws - matrix(beta, nrow(draws), length(beta), byrow = TRUE))
  max_abs <- apply(abs(deviations / se), 2, max)
  crit <- as.numeric(quantile(max_abs, 0.95))

  data.frame(
    fit = fit,
    se = se,
    lower = fit - 1.96 * se,
    upper = fit + 1.96 * se,
    simultaneous_lower = fit - crit * se,
    simultaneous_upper = fit + crit * se,
    simultaneous_crit = crit
  )
}

excluding_zero_regions <- function(ph, lower, upper) {
  flag <- ifelse(lower > 0, 1L, ifelse(upper < 0, -1L, 0L))
  runs <- rle(flag)
  ends <- cumsum(runs$lengths)
  starts <- ends - runs$lengths + 1
  keep <- runs$values != 0

  if (!any(keep)) {
    return(data.frame(ph_start = numeric(0), ph_end = numeric(0), direction = character(0)))
  }

  data.frame(
    ph_start = ph[starts[keep]],
    ph_end = ph[ends[keep]],
    direction = ifelse(runs$values[keep] > 0, "positive", "negative")
  )
}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

comparison_rows <- list()
smooth_rows <- list()
variance_rows <- list()
curve_rows <- list()
difference_rows <- list()
region_rows <- list()
kcheck_rows <- list()
concurvity_rows <- list()
strip_rows <- list()
fitted_rows <- list()

PAIRS <- t(combn(PHONES, 2))

for (spec_index in seq_len(nrow(SPECS))) {
  basis <- SPECS$basis[spec_index]
  k <- SPECS$k[spec_index]
  is_primary <- SPECS$primary[spec_index]

  for (region in REGIONS) {
    for (channel in CHANNELS) {
      response <- paste0(region, "_", channel)
      label <- sprintf("%s k=%d %s", basis, k, response)
      message("Fitting ", label)

      tag <- data.frame(
        region = region, channel = channel, basis = basis, k = k,
        primary_spec = is_primary
      )

      fits <- list()
      strip_var <- c()

      for (model_name in MODELS) {
        fit <- fit_model(response, model_name, basis, k, "REML")
        fit_ml <- fit_model(response, model_name, basis, k, "ML")
        fits[[model_name]] <- fit

        invisible(capture.output(vc <- gam.vcomp(fit)))
        vc_sd <- if (is.list(vc)) vc$all else vc[, "std.dev"]
        strip_sd <- unname(vc_sd[grep("strip_id", names(vc_sd))])
        scale_sd <- unname(vc_sd[length(vc_sd)])
        strip_var[model_name] <- strip_sd^2
        fit_summary <- summary(fit)

        comparison_rows[[length(comparison_rows) + 1]] <- cbind(tag, data.frame(
          model = model_name,
          aic_ml = AIC(fit_ml),
          reml_score = as.numeric(fit$gcv.ubre),
          deviance_explained = fit_summary$dev.expl,
          r_squared = fit_summary$r.sq,
          edf_total = sum(fit$edf),
          strip_sd = strip_sd,
          residual_sd = scale_sd,
          converged = isTRUE(fit$converged)
        ))

        s_table <- fit_summary$s.table
        smooth_rows[[length(smooth_rows) + 1]] <- cbind(tag, data.frame(
          model = model_name,
          term = rownames(s_table),
          edf = s_table[, "edf"],
          ref_df = s_table[, "Ref.df"],
          statistic = s_table[, 3],
          p_value = s_table[, 4],
          row.names = NULL
        ))

        if (model_name != "M0") {
          kc <- k.check(fit)
          kcheck_rows[[length(kcheck_rows) + 1]] <- cbind(tag, data.frame(
            model = model_name,
            term = rownames(kc),
            k_prime = kc[, "k'"],
            edf = kc[, "edf"],
            k_index = kc[, "k-index"],
            p_value = kc[, "p-value"],
            row.names = NULL
          ))

          cc <- concurvity(fit, full = TRUE)
          concurvity_rows[[length(concurvity_rows) + 1]] <- cbind(tag, data.frame(
            model = model_name,
            measure = rep(rownames(cc), times = ncol(cc)),
            term = rep(colnames(cc), each = nrow(cc)),
            value = as.vector(cc),
            row.names = NULL
          ))
        }

        if (is_primary && model_name %in% c("M1", "M2")) {
          re_coef <- coef(fit)[grep("^s\\(strip_id\\)", names(coef(fit)))]
          strip_rows[[length(strip_rows) + 1]] <- cbind(tag, data.frame(
            model = model_name,
            strip_id = levels(data$strip_id),
            strip_effect = unname(re_coef),
            row.names = NULL
          ))
        }
      }

      variance_rows[[length(variance_rows) + 1]] <- cbind(tag, data.frame(
        strip_var_M0 = strip_var[["M0"]],
        strip_var_M1 = strip_var[["M1"]],
        strip_var_M2 = strip_var[["M2"]],
        strip_var_reduction_M1_to_M2 = 1 - strip_var[["M2"]] / strip_var[["M1"]]
      ))

      # Curves and difference curves from M2.
      m2 <- fits[["M2"]]
      draws <- mvrnorm(n_sim, coef(m2), vcov(m2, unconditional = TRUE))
      X_by_phone <- lapply(PHONES, function(p) population_lp(m2, p, PH_GRID))
      names(X_by_phone) <- PHONES

      for (phone_value in PHONES) {
        intervals <- interval_frame(X_by_phone[[phone_value]], m2, draws)
        curve_rows[[length(curve_rows) + 1]] <- cbind(
          tag, data.frame(phone = phone_value, ph = PH_GRID), intervals
        )
      }

      # total: phone offset + pH-dependent part.
      # shape: pH-dependent part only (centred by-smooth difference).
      offset_columns <- grep("^phone", names(coef(m2)))

      for (pair_index in seq_len(nrow(PAIRS))) {
        first <- PAIRS[pair_index, 1]
        second <- PAIRS[pair_index, 2]
        X_total <- X_by_phone[[second]] - X_by_phone[[first]]
        X_shape <- X_total
        X_shape[, offset_columns] <- 0
        comparison <- paste(second, "minus", first)

        for (component in c("total", "shape")) {
          X_diff <- if (component == "total") X_total else X_shape
          intervals <- interval_frame(X_diff, m2, draws)

          difference_rows[[length(difference_rows) + 1]] <- cbind(
            tag,
            data.frame(comparison = comparison, component = component, ph = PH_GRID),
            intervals
          )

          for (band in c("pointwise", "simultaneous")) {
            bounds <- if (band == "pointwise") {
              list(intervals$lower, intervals$upper)
            } else {
              list(intervals$simultaneous_lower, intervals$simultaneous_upper)
            }
            found <- excluding_zero_regions(PH_GRID, bounds[[1]], bounds[[2]])
            if (nrow(found) > 0) {
              region_rows[[length(region_rows) + 1]] <- cbind(
                tag,
                data.frame(comparison = comparison, component = component, band = band),
                found
              )
            }
          }
        }
      }

      if (is_primary) {
        fitted_rows[[length(fitted_rows) + 1]] <- cbind(tag, data.frame(
          strip_id = as.character(data$strip_id),
          phone = as.character(data$phone),
          light = as.character(data$light),
          ph = data$ph,
          observed = data[[response]],
          fitted = fitted(m2),
          residual = residuals(m2, type = "response"),
          population_fit = as.vector(
            predict(m2, type = "link", exclude = "s(strip_id)")
          )
        ))
      }
    }
  }
}


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

write_rows <- function(rows, name) {
  frame <- if (length(rows) == 0) data.frame() else do.call(rbind, rows)
  write.csv(frame, file.path(output_dir, name), row.names = FALSE, fileEncoding = "UTF-8")
  invisible(frame)
}

comparison <- write_rows(comparison_rows, "model_comparison.csv")
write_rows(smooth_rows, "smooth_terms.csv")
variance <- write_rows(variance_rows, "variance_components.csv")
write_rows(curve_rows, "phone_curves.csv")
write_rows(difference_rows, "difference_curves.csv")
regions_found <- write_rows(region_rows, "difference_regions.csv")
write_rows(kcheck_rows, "k_check.csv")
write_rows(concurvity_rows, "concurvity.csv")
write_rows(strip_rows, "strip_effects.csv")
write_rows(fitted_rows, "fitted_residuals.csv")

primary_comparison <- comparison[comparison$primary_spec, ]
primary_comparison$delta_aic_vs_M1 <- NA_real_
for (i in seq_len(nrow(primary_comparison))) {
  same <- primary_comparison$region == primary_comparison$region[i] &
    primary_comparison$channel == primary_comparison$channel[i] &
    primary_comparison$model == "M1"
  primary_comparison$delta_aic_vs_M1[i] <- primary_comparison$aic_ml[i] -
    primary_comparison$aic_ml[same]
}

audit <- c(
  "GAM phone-specific pH curves (script 26)",
  strrep("=", 80),
  paste("input:", input_path),
  paste("output:", output_dir),
  paste("R:", R.version.string, "| mgcv:", as.character(packageVersion("mgcv"))),
  paste("seed:", seed, "| posterior draws:", n_sim),
  paste("phones (reference first):", paste(PHONES, collapse = ", ")),
  "",
  "Pre-specified models (method = REML; AIC from ML refits):",
  "  M0: light + phone:light + s(ph) + s(strip_id, re)",
  "  M1: M0 + phone",
  "  M2: M1 + s(ph, by = ordered phone)",
  "Primary spec: basis tp, k = 10, region contrast.",
  "Sensitivity: tp k = 6, 8, 12; cr k = 10; regions pad, bg.",
  "",
  "Primary-spec model comparison:",
  capture.output(print(
    primary_comparison[, c("region", "channel", "model", "aic_ml", "delta_aic_vs_M1",
                           "deviance_explained", "strip_sd", "residual_sd")],
    row.names = FALSE, digits = 4
  )),
  "",
  "Strip random-effect variance reduction M1 -> M2:",
  capture.output(print(
    variance[, c("region", "channel", "basis", "k", "strip_var_M1", "strip_var_M2",
                 "strip_var_reduction_M1_to_M2")],
    row.names = FALSE, digits = 4
  )),
  "",
  "Limitations:",
  "- One strip per phone x pH: the split between smooth phone curves and strip",
  "  deviations relies on the smoothness assumption.",
  "- Curves describe observed phone-specific responses, not hardware-only effects.",
  "- pH 1 and 13 are boundary points of the basis; intervals widen there.",
  "- The strip random effect can absorb any strip mean, so AIC differences",
  "  between M0-M2 are small by construction. Judge phone-specific curves by",
  "  the by-smooth tests, strip variance reduction and strip-holdout",
  "  prediction (script 27)."
)
writeLines(audit, file.path(output_dir, "analysis_audit.txt"), useBytes = TRUE)
cat(audit, sep = "\n")
