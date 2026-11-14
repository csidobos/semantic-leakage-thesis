
# ============================================================
# Semantic leakage thesis
# Analysis 07: Paired item-level bootstrap
# ============================================================
#
# Run from the repository root:
# source("thesis/analysis-07_bootstrap_32_items.R")
#
# Input:
# results/analysis/descriptive/11_full_cells_counts_and_rates.csv
#
# Default output:
# results/analysis/bootstrap_32_items/
#
# Outcomes:
#   MC: stereotype_n / n
#   MC+NAA: naa_n / n
#
# The MC+NAA stereotype rate is NOT analyzed.
#
# Resampling:
#   32 items sampled with replacement
#   All observations belonging to each sampled item retained
#   Identical bootstrap weights across all comparisons
#   10,000 repetitions by default
#   95% BCa confidence intervals
#   No multiplicity correction
#
# This is an item-level bootstrap, not the separate
# 16-social-group cluster bootstrap.
#
# Analysis tiers:
#   primary: overall MC target - neutral
#   context: other overall MC context - neutral contrasts
#   exploratory: subgroup comparisons and NAA analyses
#
# All intervals are marginal. Exploratory intervals should not
# be interpreted as multiplicity-adjusted confirmatory tests.
# ============================================================


# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

B <- suppressWarnings(
  as.integer(Sys.getenv("BOOT_B", "10000"))
)

SEED <- suppressWarnings(
  as.integer(Sys.getenv("BOOT_SEED", "20261105"))
)

ALPHA <- 0.05

if (length(B) != 1L || is.na(B) || B < 2L) {
  stop("BOOT_B must be an integer greater than or equal to 2.")
}

if (length(SEED) != 1L || is.na(SEED) ||
    SEED < 0L || SEED > .Machine$integer.max) {
  stop("BOOT_SEED must be a valid non-negative integer.")
}

CONDITIONS <- c(
  "neutral",
  "target",
  "comparison",
  "implicit",
  "nonsocial",
  "pseudo",
  "nonconnected"
)

NON_NEUTRAL <- setdiff(CONDITIONS, "neutral")

MODELS <- c("grok", "mistral", "openai", "qwen")
LANGUAGES <- c("EN", "HU")
MODES <- c("mc", "mc_unknown")
PARAPHRASES <- c("p0", "p1", "p2")
DISTANCES <- 0:2


# ------------------------------------------------------------
# 2. Paths
# ------------------------------------------------------------

find_root <- function() {

  candidates <- c(
    getwd(),
    dirname(getwd())
  )

  source_files <- lapply(sys.frames(), function(frame) {
    frame$ofile
  })

  source_files <- unlist(source_files, use.names = FALSE)

  if (length(source_files) > 0L) {
    source_dirs <- dirname(
      normalizePath(source_files, mustWork = FALSE)
    )

    candidates <- c(
      candidates,
      source_dirs,
      dirname(source_dirs)
    )
  }

  candidates <- unique(
    normalizePath(candidates, mustWork = FALSE)
  )

  for (candidate in candidates) {

    input_path <- file.path(
      candidate,
      "results",
      "analysis",
      "descriptive",
      "11_full_cells_counts_and_rates.csv"
    )

    if (file.exists(input_path)) {
      return(candidate)
    }
  }

  stop(
    "Cannot locate the repository root or input CSV. ",
    "Open C:/semantic-leakage-thesis in VS Code."
  )
}

ROOT <- find_root()

INPUT <- file.path(
  ROOT,
  "results",
  "analysis",
  "descriptive",
  "11_full_cells_counts_and_rates.csv"
)

OUT_NAME <- if (
  B == 10000L && SEED == 20261105L
) {
  "bootstrap_32_items"
} else {
  sprintf(
    "bootstrap_32_items_B%d_seed%d",
    B,
    SEED
  )
}

OUTPUT <- file.path(
  ROOT,
  "results",
  "analysis",
  OUT_NAME
)

dir.create(
  OUTPUT,
  recursive = TRUE,
  showWarnings = FALSE
)

cat("Repository:", ROOT, "\n")
cat("Bootstrap repetitions:", B, "\n")
cat("Seed:", SEED, "\n")
cat("Output:", OUTPUT, "\n\n")


# ------------------------------------------------------------
# 3. Validation helpers
# ------------------------------------------------------------

assert <- function(ok, message) {
  if (!isTRUE(ok)) {
    stop(message, call. = FALSE)
  }
}

save_csv <- function(x, filename) {

  write.csv(
    x,
    file.path(OUTPUT, filename),
    row.names = FALSE,
    na = ""
  )

  cat(
    sprintf(
      "Saved %-48s %6d rows\n",
      filename,
      nrow(x)
    )
  )

  invisible(nrow(x))
}

interval_counts <- integer()

save_intervals <- function(x, filename) {

  n <- save_csv(x, filename)

  interval_counts[filename] <<- n

  invisible(n)
}


# ------------------------------------------------------------
# 4. Read and validate input
# ------------------------------------------------------------

dat <- read.csv(
  INPUT,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

required <- c(
  "item_id",
  "group",
  "domain",
  "dimension",
  "model",
  "language",
  "response_mode",
  "condition",
  "distance",
  "paraphrase",
  "n",
  "stereotype_n",
  "naa_n",
  "content_n"
)

assert(
  all(required %in% names(dat)),
  paste(
    "Missing columns:",
    paste(setdiff(required, names(dat)), collapse = ", ")
  )
)

assert(
  nrow(dat) == 32256L,
  "Unexpected number of aggregated cells."
)

assert(
  !anyNA(dat[required]),
  "Missing values in required columns."
)

assert(
  setequal(unique(dat$condition), CONDITIONS),
  "Unexpected context conditions."
)

assert(
  setequal(unique(dat$model), MODELS),
  "Unexpected models."
)

assert(
  setequal(unique(dat$language), LANGUAGES),
  "Unexpected languages."
)

assert(
  setequal(unique(dat$response_mode), MODES),
  "Unexpected response modes."
)

assert(
  setequal(unique(dat$distance), DISTANCES),
  "Unexpected distance levels."
)

assert(
  setequal(unique(dat$paraphrase), PARAPHRASES),
  "Unexpected paraphrases."
)

assert(
  length(unique(dat$item_id)) == 32L,
  "Expected exactly 32 items."
)

assert(
  all(dat$n == 10L),
  "Expected exactly 10 repetitions per cell."
)

assert(
  all(
    dat$stereotype_n >= 0 &
    dat$naa_n >= 0 &
    dat$content_n >= 0 &
    dat$stereotype_n <= dat$content_n &
    dat$content_n + dat$naa_n == dat$n
  ),
  "Invalid response counts."
)

assert(
  all(dat$naa_n[dat$response_mode == "mc"] == 0),
  "NAA responses found in MC mode."
)

CELL_KEY <- c(
  "item_id",
  "model",
  "language",
  "response_mode",
  "condition",
  "distance",
  "paraphrase"
)

assert(
  !anyDuplicated(dat[CELL_KEY]),
  "Duplicate experimental cells."
)

EXPECTED_ROWS <- (
  32L * 4L * 2L * 2L * 7L * 3L * 3L
)

assert(
  nrow(dat) == EXPECTED_ROWS,
  "Incomplete factorial design."
)

assert(
  all(table(dat$item_id) == EXPECTED_ROWS / 32L),
  "Unbalanced item coverage."
)

assert(
  all(
    table(dat$item_id, dat$condition) ==
      4L * 2L * 2L * 3L * 3L
  ),
  "Incomplete item-condition coverage."
)

# Group names may be translated between languages.
# Each item must have one stable group within each language.

group_check <- aggregate(
  group ~ item_id + language,
  data = dat,
  FUN = function(x) length(unique(x))
)

assert(
  all(group_check$group == 1L),
  "Inconsistent item-to-group mapping."
)


# ------------------------------------------------------------
# 5. Stable item order
# ------------------------------------------------------------

ITEMS <- sort(
  unique(as.character(dat$item_id)),
  method = "radix"
)

N_ITEMS <- length(ITEMS)

dat$item_id <- factor(
  dat$item_id,
  levels = ITEMS
)

dat$model <- factor(
  dat$model,
  levels = MODELS
)

dat$language <- factor(
  dat$language,
  levels = LANGUAGES
)

dat$response_mode <- factor(
  dat$response_mode,
  levels = MODES
)

dat$condition <- factor(
  dat$condition,
  levels = CONDITIONS
)

dat$distance <- factor(
  dat$distance,
  levels = DISTANCES
)

dat$paraphrase <- factor(
  dat$paraphrase,
  levels = PARAPHRASES
)


# ------------------------------------------------------------
# 6. Paired bootstrap and jackknife weights
# ------------------------------------------------------------

set.seed(SEED)

# Each bootstrap row contains the multiplicity of every item.
# All strata use the same weights.

W <- t(
  rmultinom(
    n = B,
    size = N_ITEMS,
    prob = rep(1 / N_ITEMS, N_ITEMS)
  )
)

storage.mode(W) <- "double"

colnames(W) <- ITEMS

assert(
  all(rowSums(W) == N_ITEMS),
  "Invalid bootstrap sample sizes."
)

# Leave-one-item-out jackknife.

J <- matrix(
  1,
  nrow = N_ITEMS,
  ncol = N_ITEMS
)

diag(J) <- 0

colnames(J) <- ITEMS

FULL <- matrix(
  rep(1, N_ITEMS),
  nrow = 1
)

colnames(FULL) <- ITEMS


# ------------------------------------------------------------
# 7. BCa confidence intervals
# ------------------------------------------------------------

bca <- function(
  boot,
  jack,
  observed,
  alpha = ALPHA
) {

  if (
    !is.finite(observed) ||
    any(!is.finite(boot)) ||
    any(!is.finite(jack))
  ) {
    return(c(NA_real_, NA_real_))
  }

  if (length(unique(boot)) == 1L) {

    if (boot[1] == observed) {
      return(c(observed, observed))
    }

    return(c(NA_real_, NA_real_))
  }

  # Bias correction with half-weighted ties.

  less <- sum(boot < observed)
  ties <- sum(boot == observed)

  p0 <- (
    less + 0.5 * ties
  ) / length(boot)

  p0 <- max(
    0.5 / length(boot),
    min(
      1 - 0.5 / length(boot),
      p0
    )
  )

  z0 <- qnorm(p0)

  # Jackknife acceleration.

  delta <- mean(jack) - jack

  denominator <- 6 * (
    sum(delta^2)^(3 / 2)
  )

  if (
    !is.finite(denominator) ||
    denominator <= 0
  ) {
    return(c(NA_real_, NA_real_))
  }

  acceleration <- sum(delta^3) / denominator

  z <- qnorm(
    c(alpha / 2, 1 - alpha / 2)
  )

  correction <- 1 - acceleration * (z0 + z)

  if (
    any(!is.finite(correction)) ||
    any(abs(correction) < 1e-12)
  ) {
    return(c(NA_real_, NA_real_))
  }

  adjusted <- pnorm(
    z0 + (z0 + z) / correction
  )

  if (
    any(!is.finite(adjusted)) ||
    adjusted[1] >= adjusted[2]
  ) {
    return(c(NA_real_, NA_real_))
  }

  as.numeric(
    quantile(
      boot,
      probs = adjusted,
      names = FALSE,
      type = 7
    )
  )
}


# ------------------------------------------------------------
# 8. Build item-level count matrices
# ------------------------------------------------------------

make_cube <- function(subset_data, strata) {

  grouping_columns <- c(
    "item_id",
    strata,
    "condition"
  )

  agg <- aggregate(
    subset_data[c(
      "stereotype_n",
      "naa_n",
      "n"
    )],
    by = subset_data[grouping_columns],
    FUN = sum
  )

  if (length(strata) == 0L) {

    agg$.stratum <- "all"

  } else {

    agg$.stratum <- do.call(
      paste,
      c(agg[strata], sep = "|")
    )
  }

  strata_labels <- unique(agg$.stratum)

  # Explicit stratum-major ordering.

  keys <- unlist(
    lapply(
      strata_labels,
      function(s) paste(s, CONDITIONS, sep = "::")
    ),
    use.names = FALSE
  )

  cell_keys <- paste(
    agg$.stratum,
    agg$condition,
    sep = "::"
  )

  column_idx <- match(cell_keys, keys)

  row_idx <- match(
    as.character(agg$item_id),
    ITEMS
  )

  assert(
    !anyNA(column_idx) && !anyNA(row_idx),
    "Invalid item-stratum mapping."
  )

  assert(
    !anyDuplicated(
      data.frame(row_idx, column_idx)
    ),
    "Duplicate item-stratum-condition cells."
  )

  assert(
    nrow(agg) == N_ITEMS * length(keys),
    "Missing item-stratum-condition cells."
  )

  measures <- c(
    "stereotype_n",
    "naa_n",
    "n"
  )

  matrices <- lapply(measures, function(measure) {

    out <- matrix(
      NA_real_,
      nrow = N_ITEMS,
      ncol = length(keys),
      dimnames = list(ITEMS, keys)
    )

    out[cbind(row_idx, column_idx)] <- agg[[measure]]

    assert(
      !anyNA(out),
      paste("Missing count values:", measure)
    )

    out
  })

  names(matrices) <- measures

  list(
    mats = matrices,
    labels = strata_labels,
    keys = keys
  )
}


# ------------------------------------------------------------
# 9. Calculate weighted response rates
# ------------------------------------------------------------

calculate_rates <- function(cube, metric) {

  numerator <- switch(
    metric,
    stereotype = cube$mats$stereotype_n,
    naa = cube$mats$naa_n,
    stop("Unknown metric.")
  )

  denominator <- cube$mats$n

  calculate <- function(weights) {

    a <- weights %*% numerator
    d <- weights %*% denominator

    out <- 100 * a / d

    out[d == 0] <- NA_real_

    out
  }

  list(
    estimate = calculate(FULL),
    boot = calculate(W),
    jack = calculate(J)
  )
}


# ------------------------------------------------------------
# 10. Paired condition-minus-neutral contrasts
# ------------------------------------------------------------

context_effects <- function(rates, cube) {

  labels <- cube$labels

  condition_keys <- unlist(
    lapply(
      labels,
      function(s) {
        paste(s, NON_NEUTRAL, sep = "::")
      }
    ),
    use.names = FALSE
  )

  neutral_keys <- rep(
    paste(labels, "neutral", sep = "::"),
    each = length(NON_NEUTRAL)
  )

  idx_a <- match(condition_keys, cube$keys)
  idx_n <- match(neutral_keys, cube$keys)

  assert(
    !anyNA(idx_a) && !anyNA(idx_n),
    "Missing neutral pairing."
  )

  list(
    labels = rep(
      labels,
      each = length(NON_NEUTRAL)
    ),
    condition = rep(
      NON_NEUTRAL,
      times = length(labels)
    ),
    estimate =
      rates$estimate[, idx_a, drop = FALSE] -
      rates$estimate[, idx_n, drop = FALSE],
    boot =
      rates$boot[, idx_a, drop = FALSE] -
      rates$boot[, idx_n, drop = FALSE],
    jack =
      rates$jack[, idx_a, drop = FALSE] -
      rates$jack[, idx_n, drop = FALSE]
  )
}


# ------------------------------------------------------------
# 11. Summarize BCa estimates
# ------------------------------------------------------------

summarize_statistics <- function(
  estimate,
  boot,
  jack
) {

  observed <- as.numeric(estimate)

  assert(
    ncol(boot) == length(observed),
    "Bootstrap dimension mismatch."
  )

  assert(
    ncol(jack) == length(observed),
    "Jackknife dimension mismatch."
  )

  intervals <- t(
    vapply(
      seq_along(observed),
      function(k) {
        bca(
          boot[, k],
          jack[, k],
          observed[k]
        )
      },
      numeric(2)
    )
  )

  data.frame(
    estimate_pp = observed,
    ci_lower = intervals[, 1],
    ci_upper = intervals[, 2],
    bca_defined =
      is.finite(intervals[, 1]) &
      is.finite(intervals[, 2]),
    ci_zero_width =
      is.finite(intervals[, 1]) &
      is.finite(intervals[, 2]) &
      intervals[, 1] == intervals[, 2]
  )
}


# ------------------------------------------------------------
# 12. Grouping definitions
# ------------------------------------------------------------

GROUPINGS <- list(
  overall = character(),
  model = "model",
  language = "language",
  distance = "distance",
  paraphrase = "paraphrase",
  model_language = c(
    "model", "language"
  ),
  model_distance = c(
    "model", "distance"
  ),
  model_paraphrase = c(
    "model", "paraphrase"
  ),
  language_distance = c(
    "language", "distance"
  ),
  language_paraphrase = c(
    "language", "paraphrase"
  ),
  model_language_distance = c(
    "model", "language", "distance"
  ),
  model_language_paraphrase = c(
    "model", "language", "paraphrase"
  )
)


# ------------------------------------------------------------
# 13. Run one grouped analysis
# ------------------------------------------------------------

run_grouping <- function(
  name,
  grouping,
  mode,
  metric,
  calculate_intervals = TRUE
) {

  subset_data <- dat[
    dat$response_mode == mode,
    ,
    drop = FALSE
  ]

  cube <- make_cube(
    subset_data,
    grouping
  )

  rates <- calculate_rates(
    cube,
    metric
  )

  effects <- context_effects(
    rates,
    cube
  )

  # Return raw paired estimates for comparisons without
  # calculating unused intervals.

  if (!calculate_intervals) {

    return(list(
      raw_effects = effects,
      raw_rates = rates,
      cube = cube
    ))
  }

  n_strata <- length(cube$labels)

  rate_info <- data.frame(
    stratum = rep(
      cube$labels,
      each = length(CONDITIONS)
    ),
    condition = rep(
      CONDITIONS,
      times = n_strata
    )
  )

  rate_keys <- paste(
    rate_info$stratum,
    rate_info$condition,
    sep = "::"
  )

  rate_idx <- match(
    rate_keys,
    cube$keys
  )

  assert(
    !anyNA(rate_idx),
    "Missing rate export keys."
  )

  rate_stats <- summarize_statistics(
    rates$estimate[, rate_idx, drop = FALSE],
    rates$boot[, rate_idx, drop = FALSE],
    rates$jack[, rate_idx, drop = FALSE]
  )

  rate_out <- cbind(
    data.frame(
      analysis = rep(name, nrow(rate_info)),
      metric = rep(metric, nrow(rate_info)),
      mode = rep(mode, nrow(rate_info))
    ),
    rate_info,
    rate_stats
  )

  effect_info <- data.frame(
    stratum = effects$labels,
    condition = effects$condition,
    reference = "neutral"
  )

  effect_stats <- summarize_statistics(
    effects$estimate,
    effects$boot,
    effects$jack
  )

  effect_out <- cbind(
    data.frame(
      analysis = rep(name, nrow(effect_info)),
      metric = rep(metric, nrow(effect_info)),
      mode = rep(mode, nrow(effect_info))
    ),
    effect_info,
    effect_stats
  )

  assert(
    nrow(rate_out) ==
      n_strata * length(CONDITIONS),
    "Incomplete context rate output."
  )

  assert(
    nrow(effect_out) ==
      n_strata * length(NON_NEUTRAL),
    "Incomplete neutral contrast output."
  )

  list(
    rates = rate_out,
    effects = effect_out,
    raw_effects = effects,
    raw_rates = rates,
    cube = cube
  )
}


# ------------------------------------------------------------
# 14. MC stereotype analysis
# ------------------------------------------------------------

all_rates <- list()
all_effects <- list()

overall_mc_result <- NULL

for (name in names(GROUPINGS)) {

  cat("Running MC stereotype:", name, "\n")

  result <- run_grouping(
    name = name,
    grouping = GROUPINGS[[name]],
    mode = "mc",
    metric = "stereotype"
  )

  all_rates[[name]] <- result$rates
  all_effects[[name]] <- result$effects

  if (name == "overall") {
    overall_mc_result <- result
  }
}

mc_rates <- do.call(
  rbind,
  all_rates
)

mc_rates$tier <- "descriptive"

save_intervals(
  mc_rates,
  "01_context_rates_bca.csv"
)

mc_effects <- do.call(
  rbind,
  all_effects
)

mc_effects$tier <- ifelse(
  mc_effects$analysis == "overall" &
    mc_effects$condition == "target",
  "primary",
  ifelse(
    mc_effects$analysis == "overall",
    "context",
    "exploratory"
  )
)

assert(
  sum(mc_effects$tier == "primary") == 1L,
  "Expected exactly one primary effect."
)

save_intervals(
  mc_effects,
  "02_neutral_contrasts_bca.csv"
)


# ------------------------------------------------------------
# 15. Between-stratum comparisons for MC
# ------------------------------------------------------------

# Each comparison is a difference between two paired
# condition-minus-neutral effects.
#
# Example:
# (Target - Neutral in HU) -
# (Target - Neutral in EN)
#
# Identical item bootstrap weights are used in both parts.

PAIR_SPECS <- list(
  model = MODELS,
  language = LANGUAGES,
  distance = as.character(DISTANCES),
  paraphrase = PARAPHRASES
)

comparison_results <- list()

comparison_counter <- 0L

for (variable in names(PAIR_SPECS)) {

  cat("Running paired comparisons:", variable, "\n")

  levels_to_compare <- PAIR_SPECS[[variable]]

  result <- run_grouping(
    name = paste0("comparison_", variable),
    grouping = variable,
    mode = "mc",
    metric = "stereotype",
    calculate_intervals = FALSE
  )

  effects <- result$raw_effects

  pairs <- combn(
    levels_to_compare,
    2,
    simplify = FALSE
  )

  effect_keys <- paste(
    effects$labels,
    effects$condition,
    sep = "::"
  )

  for (condition in NON_NEUTRAL) {

    for (pair in pairs) {

      key_a <- paste(
        pair[1],
        condition,
        sep = "::"
      )

      key_b <- paste(
        pair[2],
        condition,
        sep = "::"
      )

      idx_a <- match(
        key_a,
        effect_keys
      )

      idx_b <- match(
        key_b,
        effect_keys
      )

      assert(
        !is.na(idx_a) && !is.na(idx_b),
        "Missing paired comparison."
      )

      estimate <-
        effects$estimate[, idx_a] -
        effects$estimate[, idx_b]

      boot <-
        effects$boot[, idx_a] -
        effects$boot[, idx_b]

      jack <-
        effects$jack[, idx_a] -
        effects$jack[, idx_b]

      stats <- summarize_statistics(
        matrix(estimate, ncol = 1),
        matrix(boot, ncol = 1),
        matrix(jack, ncol = 1)
      )

      comparison_counter <- comparison_counter + 1L

      comparison_results[[comparison_counter]] <- cbind(
        data.frame(
          variable = variable,
          mode = "mc",
          metric = "stereotype",
          condition = condition,
          reference = "neutral",
          level_a = pair[1],
          level_b = pair[2],
          comparison = "effect_a_minus_effect_b",
          tier = "exploratory"
        ),
        stats
      )
    }
  }
}

between_strata <- do.call(
  rbind,
  comparison_results
)

assert(
  nrow(between_strata) == 78L,
  "Unexpected number of between-stratum comparisons."
)

save_intervals(
  between_strata,
  "03_between_strata_effect_differences_bca.csv"
)


# ------------------------------------------------------------
# 16. MC+NAA selection-rate analysis
# ------------------------------------------------------------

# Only naa_n / n is evaluated in MC+NAA mode.
# No stereotype rate is calculated for this response mode.

naa_rates <- list()
naa_effects <- list()

overall_naa_result <- NULL

for (name in names(GROUPINGS)) {

  cat("Running MC+NAA selection:", name, "\n")

  result <- run_grouping(
    name = name,
    grouping = GROUPINGS[[name]],
    mode = "mc_unknown",
    metric = "naa"
  )

  result$rates$tier <- "exploratory"
  result$effects$tier <- "exploratory"

  naa_rates[[name]] <- result$rates
  naa_effects[[name]] <- result$effects

  if (name == "overall") {
    overall_naa_result <- result
  }
}

save_intervals(
  do.call(rbind, naa_rates),
  "04_naa_rates_bca.csv"
)

save_intervals(
  do.call(rbind, naa_effects),
  "05_naa_neutral_contrasts_bca.csv"
)


# ------------------------------------------------------------
# 17. Item-level descriptive summaries
# ------------------------------------------------------------

item_columns <- c(
  "item_id",
  "model",
  "language",
  "response_mode",
  "condition"
)

item_agg <- aggregate(
  dat[c(
    "stereotype_n",
    "naa_n",
    "n"
  )],
  by = dat[item_columns],
  FUN = sum
)

item_agg$metric <- ifelse(
  as.character(item_agg$response_mode) == "mc",
  "stereotype",
  "naa"
)

item_agg$numerator <- ifelse(
  item_agg$metric == "stereotype",
  item_agg$stereotype_n,
  item_agg$naa_n
)

item_agg$value_pct <- (
  100 * item_agg$numerator / item_agg$n
)

item_agg <- item_agg[c(
  item_columns,
  "metric",
  "numerator",
  "n",
  "value_pct"
)]

assert(
  nrow(item_agg) == 3584L,
  "Unexpected item-level descriptive row count."
)

save_csv(
  item_agg,
  "06_item_context_descriptive.csv"
)


# ------------------------------------------------------------
# 18. Social-group descriptive summaries
# ------------------------------------------------------------

# Identify the two items belonging to each social group.
# The item_id suffix is removed only after validation.

item_to_group <- unique(
  dat[c("item_id", "language", "group")]
)

item_to_group$item_id <- as.character(
  item_to_group$item_id
)

# Derive a language-independent identifier from item_id.
# This assumes item IDs have a final item-specific suffix.

item_to_group$social_group_id <- sub(
  "_[^_]+$",
  "",
  item_to_group$item_id
)

item_mapping <- unique(
  item_to_group[c("item_id", "social_group_id")]
)

assert(
  nrow(item_mapping) == N_ITEMS,
  "Each item must have exactly one social-group ID."
)

assert(
  length(unique(item_mapping$social_group_id)) == 16L,
  "Expected exactly 16 social groups."
)

assert(
  all(table(item_mapping$social_group_id) == 2L),
  "Expected two items per social group."
)

item_agg$item_id <- as.character(
  item_agg$item_id
)

item_agg <- merge(
  item_agg,
  item_mapping,
  by = "item_id",
  all.x = TRUE,
  sort = FALSE
)

assert(
  nrow(item_agg) == 3584L &&
    !anyNA(item_agg$social_group_id),
  "Invalid social-group merge."
)

group_columns <- c(
  "social_group_id",
  "model",
  "language",
  "response_mode",
  "condition",
  "metric"
)

group_agg <- aggregate(
  item_agg[c("numerator", "n")],
  by = item_agg[group_columns],
  FUN = sum
)

group_agg$value_pct <- (
  100 * group_agg$numerator / group_agg$n
)

assert(
  nrow(group_agg) == 1792L,
  "Unexpected social-group descriptive row count."
)

save_csv(
  group_agg,
  "07_social_group_context_descriptive.csv"
)


# ------------------------------------------------------------
# 19. Descriptive neutral contrasts
# ------------------------------------------------------------

describe_neutral <- function(x, id_columns) {

  other <- x[
    x$condition != "neutral",
    c(id_columns, "condition", "value_pct"),
    drop = FALSE
  ]

  neutral <- x[
    x$condition == "neutral",
    c(id_columns, "value_pct"),
    drop = FALSE
  ]

  names(neutral)[ncol(neutral)] <- "neutral_pct"

  assert(
    !anyDuplicated(neutral[id_columns]),
    "Duplicate neutral descriptive keys."
  )

  out <- merge(
    other,
    neutral,
    by = id_columns,
    all.x = TRUE,
    sort = FALSE
  )

  assert(
    nrow(out) == nrow(other) &&
      !anyNA(out$neutral_pct),
    "Missing or duplicated neutral matches."
  )

  out$delta_pp <- (
    out$value_pct - out$neutral_pct
  )

  out
}

item_neutral <- describe_neutral(
  item_agg,
  c(
    "item_id",
    "model",
    "language",
    "response_mode",
    "metric"
  )
)

assert(
  nrow(item_neutral) == 3072L,
  "Unexpected item neutral contrast count."
)

save_csv(
  item_neutral,
  "08_item_neutral_contrasts_descriptive.csv"
)

group_neutral <- describe_neutral(
  group_agg,
  c(
    "social_group_id",
    "model",
    "language",
    "response_mode",
    "metric"
  )
)

assert(
  nrow(group_neutral) == 1536L,
  "Unexpected social-group neutral contrast count."
)

save_csv(
  group_neutral,
  "09_social_group_neutral_contrasts_descriptive.csv"
)


# ------------------------------------------------------------
# 20. Validation of overall MC results
# ------------------------------------------------------------

overall_mc <- item_agg[
  as.character(item_agg$response_mode) == "mc",
  ,
  drop = FALSE
]

raw_mc <- aggregate(
  cbind(numerator, n) ~ condition,
  data = overall_mc,
  FUN = sum
)

raw_mc$pct <- (
  100 * raw_mc$numerator / raw_mc$n
)

for (condition in CONDITIONS) {

  calculated <- overall_mc_result$rates$estimate_pp[
    overall_mc_result$rates$condition == condition
  ]

  expected <- raw_mc$pct[
    raw_mc$condition == condition
  ]

  assert(
    length(calculated) == 1L &&
      length(expected) == 1L &&
      abs(calculated - expected) < 1e-9,
    paste("MC rate mismatch:", condition)
  )
}

assert(
  nrow(overall_mc_result$effects) == 6L,
  "Expected six overall MC neutral contrasts."
)

assert(
  setequal(
    overall_mc_result$effects$condition,
    NON_NEUTRAL
  ),
  "Missing overall MC neutral contrast."
)


# ------------------------------------------------------------
# 21. Validation of overall NAA results
# ------------------------------------------------------------

overall_naa <- item_agg[
  as.character(item_agg$response_mode) == "mc_unknown",
  ,
  drop = FALSE
]

raw_naa <- aggregate(
  cbind(numerator, n) ~ condition,
  data = overall_naa,
  FUN = sum
)

raw_naa$pct <- (
  100 * raw_naa$numerator / raw_naa$n
)

for (condition in CONDITIONS) {

  calculated <- overall_naa_result$rates$estimate_pp[
    overall_naa_result$rates$condition == condition
  ]

  expected <- raw_naa$pct[
    raw_naa$condition == condition
  ]

  assert(
    length(calculated) == 1L &&
      length(expected) == 1L &&
      abs(calculated - expected) < 1e-9,
    paste("NAA rate mismatch:", condition)
  )
}


# ------------------------------------------------------------
# 22. Algebraic validation of paired contrasts
# ------------------------------------------------------------

# For each resample:
# Target - Neutral =
# (Target - Implicit) + (Implicit - Neutral)

mc_cube <- overall_mc_result$cube
mc_raw_rates <- overall_mc_result$raw_rates

target_idx <- match(
  "all::target",
  mc_cube$keys
)

implicit_idx <- match(
  "all::implicit",
  mc_cube$keys
)

neutral_idx <- match(
  "all::neutral",
  mc_cube$keys
)

direct_effect <- (
  mc_raw_rates$boot[, target_idx] -
  mc_raw_rates$boot[, neutral_idx]
)

decomposed_effect <- (
  mc_raw_rates$boot[, target_idx] -
  mc_raw_rates$boot[, implicit_idx]
) + (
  mc_raw_rates$boot[, implicit_idx] -
  mc_raw_rates$boot[, neutral_idx]
)

assert(
  max(abs(direct_effect - decomposed_effect)) < 1e-9,
  "Paired contrast algebra failed."
)


# ------------------------------------------------------------
# 23. Interval diagnostics
# ------------------------------------------------------------

interval_files <- c(
  "01_context_rates_bca.csv",
  "02_neutral_contrasts_bca.csv",
  "03_between_strata_effect_differences_bca.csv",
  "04_naa_rates_bca.csv",
  "05_naa_neutral_contrasts_bca.csv"
)

diagnostics <- lapply(interval_files, function(filename) {

  x <- read.csv(
    file.path(OUTPUT, filename),
    stringsAsFactors = FALSE
  )

  data.frame(
    file = filename,
    n_intervals = nrow(x),
    n_undefined = sum(!x$bca_defined),
    n_zero_width = sum(x$ci_zero_width, na.rm = TRUE)
  )
})

diagnostics <- do.call(
  rbind,
  diagnostics
)

save_csv(
  diagnostics,
  "12_interval_counts.csv"
)


# ------------------------------------------------------------
# 24. Reproducibility settings
# ------------------------------------------------------------

settings <- data.frame(
  setting = c(
    "seed",
    "bootstrap_repetitions",
    "n_items",
    "n_conditions",
    "n_models",
    "n_languages",
    "n_distances",
    "n_paraphrases",
    "interval",
    "resampling_unit",
    "alpha",
    "n_exported_intervals",
    "multiplicity_correction",
    "mc_outcome",
    "mc_unknown_outcome",
    "r_version",
    "rng_kind"
  ),
  value = c(
    SEED,
    B,
    N_ITEMS,
    length(CONDITIONS),
    length(MODELS),
    length(LANGUAGES),
    length(DISTANCES),
    length(PARAPHRASES),
    "BCa 95%",
    "item_id",
    ALPHA,
    sum(diagnostics$n_intervals),
    "none",
    "stereotype_n / n",
    "naa_n / n",
    R.version.string,
    paste(RNGkind(), collapse = "/")
  )
)

save_csv(
  settings,
  "10_analysis_settings.csv"
)

writeLines(
  capture.output(sessionInfo()),
  file.path(
    OUTPUT,
    "11_session_info.txt"
  )
)


# ------------------------------------------------------------
# 25. Final checks and summary
# ------------------------------------------------------------

primary <- mc_effects[
  mc_effects$tier == "primary",
  ,
  drop = FALSE
]

assert(
  nrow(primary) == 1L,
  "Primary effect was not uniquely identified."
)

cat("\n")
cat("============================================\n")
cat("BOOTSTRAP FINISHED\n")
cat("============================================\n")

cat(
  "Primary MC Target - Neutral estimate:",
  round(primary$estimate_pp, 4),
  "pp\n"
)

cat(
  "95% BCa CI:",
  round(primary$ci_lower, 4),
  "to",
  round(primary$ci_upper, 4),
  "pp\n"
)

cat(
  "Exported intervals:",
  sum(diagnostics$n_intervals),
  "\n"
)

cat(
  "Undefined BCa intervals:",
  sum(diagnostics$n_undefined),
  "\n"
)

cat(
  "Zero-width BCa intervals:",
  sum(diagnostics$n_zero_width),
  "\n"
)

cat(
  "Output directory:",
  OUTPUT,
  "\n"
)

cat("============================================\n")
