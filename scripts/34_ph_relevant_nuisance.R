#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
#
# 34_ph_relevant_nuisance.R
#
# How much does each nuisance component move the pH reading?
#
# A colour shift v at pH x changes the inverse pH reading by approximately
#
#     delta_pH(x) = (v . t_hat(x)) / s(x)
#
# where t(x) = dT/dpH is the tangent of the pH colour trajectory T(x),
# t_hat its unit vector and s(x) = |t(x)| the colour speed (Delta E per pH unit).
# Only the component of v along the trajectory matters; the orthogonal part
# changes the colour but not the reading.
#
# Components (from the script 26 M2 fit, tp k = 10, strip effect = 0)
# -------------------------------------------------------------------
#   T(x)            mean of the three phone curves, averaged over lights
#   phone offset    o_p = mean over the pH grid of (curve_p(x) - T(x))
#   phone shape     g_p(x) = curve_p(x) - T(x) - o_p
#   light           curve_{p,l}(x) - curve_p(x)   (light + phone:light)
#   strip noise     non-smooth strip SD per channel (random effect)
#   image noise     residual SD per channel
#
# For vector components the signed pH shift is reported; for the noise terms
# the SD of the pH shift, sqrt(sum_j (t_hat_j sigma_j)^2) / s.
#
# Alignment: for the phone offset and shape, the angle between the colour shift
# and the pH direction, acos(|v . t_hat| / |v|), in 0-90 degrees (0 = along the
# pH direction, 90 = orthogonal). Summarised per draw as the median and mean
# over pH 2-12.
#
# Uncertainty: 1000 draws from the posterior of the M2 coefficients
# (Vp, unconditional), applied to every derived quantity.
#
# Outputs
# -------
# outputs/ph_relevant_nuisance/
#   component_shift_by_ph.csv   region x phone x component x light x pH (median, 95% interval)
#   component_summary.csv       mean |pH shift| over pH 2-12 and pH 1/13 (with intervals)
#   trajectory.csv              T(x), colour speed s(x)
#   component_angle.csv         offset/shape angle to the pH direction over pH 2-12
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
output_dir <- resolve(get_arg("output-dir", "outputs/ph_relevant_nuisance"))
n_draws <- as.integer(get_arg("n-draws", "1000"))
seed <- as.integer(get_arg("seed", "20261009"))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

REGIONS <- c("contrast", "pad")
CHANNELS <- c("L", "a", "b")
K <- 10L
BASIS <- "tp"
PH_GRID <- seq(1, 13, by = 0.05)
INTERIOR <- PH_GRID >= 2 & PH_GRID <= 12
BOUNDARY <- PH_GRID <= 1.5 | PH_GRID >= 12.5


# ---------------------------------------------------------------------------
# Data and model (same as script 26)
# ---------------------------------------------------------------------------

raw <- read.csv(input_path, fileEncoding = "UTF-8-BOM", stringsAsFactors = FALSE)

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

m2_formula <- function(response) {
  fixed <- paste(c("phone", "light", PL_NAMES), collapse = " + ")
  as.formula(paste(
    response, "~", fixed,
    sprintf("+ s(ph, bs = '%s', k = %d)", BASIS, K),
    sprintf("+ s(ph, by = phone_ord, bs = '%s', k = %d)", BASIS, K),
    "+ s(strip_id, bs = 're')"
  ))
}

grid_frame <- function(phone_value, light_value) {
  grid <- data.frame(
    ph = PH_GRID,
    phone = factor(phone_value, levels = PHONES),
    phone_ord = factor(phone_value, levels = PHONES, ordered = TRUE),
    light = factor(light_value, levels = LIGHTS),
    strip_id = factor(levels(data$strip_id)[1], levels = levels(data$strip_id))
  )
  cbind(grid, pl_matrix(grid$phone, grid$light))
}

strip_sd <- function(fit) {
  invisible(capture.output(vc <- gam.vcomp(fit)))
  sd <- if (is.list(vc)) vc$all else vc[, "std.dev"]
  unname(sd[grep("strip_id", names(sd))])
}


# ---------------------------------------------------------------------------
# Curves for every draw: array [grid, draw, channel] per phone x light
# ---------------------------------------------------------------------------

quantiles <- function(matrix_by_draw) {
  # rows = grid points (or 1), columns = draws
  t(apply(matrix_by_draw, 1, quantile, probs = c(0.5, 0.025, 0.975)))
}

shift_rows <- list()
summary_rows <- list()
trajectory_rows <- list()
angle_rows <- list()
audit_rows <- list()

for (region in REGIONS) {
  curves <- list()          # curves[[phone]][[light]] = array [grid, draw, channel]
  sigma_strip <- c()
  sigma_image <- c()

  for (channel in CHANNELS) {
    fit <- gam(m2_formula(paste0(region, "_", channel)), data = data, method = "REML")
    beta_draws <- mvrnorm(n_draws, coef(fit), vcov(fit, unconditional = TRUE))
    sigma_strip[channel] <- strip_sd(fit)
    sigma_image[channel] <- sqrt(fit$sig2)

    for (phone_value in PHONES) {
      for (light_value in LIGHTS) {
        X <- predict(fit, newdata = grid_frame(phone_value, light_value), type = "lpmatrix")
        X[, grep("^s\\(strip_id\\)", colnames(X))] <- 0
        values <- X %*% t(beta_draws)                       # grid x draws
        key <- paste(phone_value, light_value)
        if (is.null(curves[[key]])) {
          curves[[key]] <- array(NA_real_, c(length(PH_GRID), n_draws, 3))
        }
        curves[[key]][, , match(channel, CHANNELS)] <- values
      }
    }
  }

  phone_curve <- lapply(PHONES, function(p) {
    Reduce(`+`, lapply(LIGHTS, function(l) curves[[paste(p, l)]])) / length(LIGHTS)
  })
  names(phone_curve) <- PHONES
  trajectory <- Reduce(`+`, phone_curve) / length(PHONES)   # [grid, draw, channel]

  # Tangent by central differences along the grid.
  tangent <- array(NA_real_, dim(trajectory))
  for (channel_index in 1:3) {
    for (draw in seq_len(n_draws)) {
      tangent[, draw, channel_index] <- c(
        diff(trajectory[1:2, draw, channel_index]),
        (trajectory[3:length(PH_GRID), draw, channel_index] -
           trajectory[1:(length(PH_GRID) - 2), draw, channel_index]) / 2,
        diff(trajectory[(length(PH_GRID) - 1):length(PH_GRID), draw, channel_index])
      ) / diff(PH_GRID[1:2])
    }
  }
  speed <- sqrt(apply(tangent^2, c(1, 2), sum))              # [grid, draw]
  unit <- tangent / array(speed, dim(tangent))

  ph_shift <- function(vector_field) {
    # signed pH shift: (v . t_hat) / s ; vector_field [grid, draw, channel]
    apply(vector_field * unit, c(1, 2), sum) / speed
  }

  add_angle <- function(phone_value, component, vector_field) {
    # acute angle (degrees) between the shift and the pH direction, [grid, draw]
    along <- abs(apply(vector_field * unit, c(1, 2), sum))
    size <- sqrt(apply(vector_field^2, c(1, 2), sum))
    angle <- acos(pmin(along / size, 1)) * 180 / pi
    for (statistic in c("median", "mean")) {
      per_draw <- apply(angle[INTERIOR, , drop = FALSE], 2, statistic)
      angle_rows[[length(angle_rows) + 1]] <<- data.frame(
        region = region, phone = phone_value, component = component,
        statistic = paste0(statistic, "_over_ph_2_12"),
        angle_median = median(per_draw),
        angle_lower = unname(quantile(per_draw, 0.025)),
        angle_upper = unname(quantile(per_draw, 0.975))
      )
    }
  }

  speed_q <- quantiles(speed)
  trajectory_rows[[region]] <- data.frame(
    region = region, ph = PH_GRID,
    T_L = apply(trajectory[, , 1], 1, median),
    T_a = apply(trajectory[, , 2], 1, median),
    T_b = apply(trajectory[, , 3], 1, median),
    speed_median = speed_q[, 1], speed_lower = speed_q[, 2], speed_upper = speed_q[, 3]
  )

  add_component <- function(phone_value, component, light_value, shift) {
    q <- quantiles(shift)
    shift_rows[[length(shift_rows) + 1]] <<- data.frame(
      region = region, phone = phone_value, component = component, light = light_value,
      ph = PH_GRID, shift_median = q[, 1], shift_lower = q[, 2], shift_upper = q[, 3]
    )
    for (subset in c("interior_ph_2_12", "boundary_ph_1_13")) {
      rows <- if (subset == "interior_ph_2_12") INTERIOR else BOUNDARY
      mean_abs <- colMeans(abs(shift[rows, , drop = FALSE]))
      summary_rows[[length(summary_rows) + 1]] <<- data.frame(
        region = region, phone = phone_value, component = component, light = light_value,
        subset = subset,
        mean_abs_shift_median = median(mean_abs),
        mean_abs_shift_lower = unname(quantile(mean_abs, 0.025)),
        mean_abs_shift_upper = unname(quantile(mean_abs, 0.975))
      )
    }
  }

  for (phone_value in PHONES) {
    deviation <- phone_curve[[phone_value]] - trajectory
    offset <- apply(deviation, c(2, 3), mean)                 # [draw, channel]
    offset_field <- array(rep(offset, each = length(PH_GRID)), dim(deviation))
    shape_field <- deviation - offset_field

    add_component(phone_value, "phone_offset", "all", ph_shift(offset_field))
    add_component(phone_value, "phone_shape", "all", ph_shift(shape_field))
    add_component(phone_value, "phone_total", "all", ph_shift(deviation))
    add_angle(phone_value, "phone_offset", offset_field)
    add_angle(phone_value, "phone_shape", shape_field)

    for (light_value in LIGHTS) {
      light_field <- curves[[paste(phone_value, light_value)]] - phone_curve[[phone_value]]
      add_component(phone_value, "light", light_value, ph_shift(light_field))
    }
  }

  for (noise in c("strip_noise", "image_noise")) {
    sigma <- if (noise == "strip_noise") sigma_strip else sigma_image
    along <- sqrt(apply(unit^2 * array(rep(sigma^2, each = length(PH_GRID) * n_draws),
                                       dim(unit)), c(1, 2), sum))
    add_component("all", noise, "all", along / speed)
  }

  audit_rows[[region]] <- data.frame(
    region = region,
    sigma_strip_L = sigma_strip[["L"]], sigma_strip_a = sigma_strip[["a"]],
    sigma_strip_b = sigma_strip[["b"]],
    sigma_image_L = sigma_image[["L"]], sigma_image_a = sigma_image[["a"]],
    sigma_image_b = sigma_image[["b"]]
  )
}

write.csv(do.call(rbind, shift_rows), file.path(output_dir, "component_shift_by_ph.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
summary <- do.call(rbind, summary_rows)
write.csv(summary, file.path(output_dir, "component_summary.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
write.csv(do.call(rbind, trajectory_rows), file.path(output_dir, "trajectory.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
angles <- do.call(rbind, angle_rows)
write.csv(angles, file.path(output_dir, "component_angle.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

view <- summary[summary$light %in% c("all") | summary$component == "light", ]
light_max <- aggregate(mean_abs_shift_median ~ region + phone + subset,
                       data = summary[summary$component == "light", ], FUN = max)

notes <- c(
  "pH-relevant nuisance decomposition (script 34)",
  strrep("=", 80),
  paste("R:", R.version.string, "| mgcv:", as.character(packageVersion("mgcv"))),
  paste("model: M2, tp k = 10; posterior draws:", n_draws, "| seed:", seed),
  "pH shift = (colour shift . unit tangent) / colour speed; noise terms give SD",
  "",
  "Mean |pH shift| over pH 2-12 (median and 95% interval over draws):",
  capture.output(print(
    summary[summary$subset == "interior_ph_2_12" & summary$component != "light",
            c("region", "phone", "component", "mean_abs_shift_median",
              "mean_abs_shift_lower", "mean_abs_shift_upper")],
    row.names = FALSE, digits = 3
  )),
  "",
  "Largest light component per phone (mean |pH shift|, median over draws):",
  capture.output(print(light_max, row.names = FALSE, digits = 3)),
  "",
  "Angle to the pH direction, pH 2-12 (0 = aligned, 90 = orthogonal; median and 95% interval over draws):",
  capture.output(print(angles, row.names = FALSE, digits = 3)),
  "",
  "Noise SDs used:",
  capture.output(print(do.call(rbind, audit_rows), row.names = FALSE, digits = 3))
)
writeLines(notes, file.path(output_dir, "analysis_audit.txt"), useBytes = TRUE)
cat(notes, sep = "\n")
