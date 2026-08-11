#!/usr/bin/env Rscript

# Regenerate scalar evidence from a pinned R/mice numeric FCS run. The output
# deliberately records aggregate metrics only: it never serialises input rows
# or completed datasets. A later reviewed conformance fixture may use this
# evidence, but this script by itself makes no parity assertion about Missingly.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: generate_mice_reference.R <numeric_fcs.csv> <output.json>")
}

input_path <- args[[1L]]
output_path <- args[[2L]]
if (!file.exists(input_path)) {
  stop("Input fixture does not exist: ", input_path)
}

for (package_name in c("jsonlite", "mice")) {
  if (!requireNamespace(package_name, quietly = TRUE)) {
    stop("Required package is unavailable: ", package_name)
  }
}

fixture <- utils::read.csv(input_path, check.names = FALSE, na.strings = c("", "NA"))
if (!all(vapply(fixture, is.numeric, logical(1L)))) {
  stop("The numeric FCS fixture must contain numeric columns only")
}

seed <- 20260811L
imputation_count <- 3L
iteration_count <- 5L
methods <- stats::setNames(rep("norm", ncol(fixture)), names(fixture))
fit <- mice::mice(
  fixture,
  m = imputation_count,
  maxit = iteration_count,
  method = methods,
  printFlag = FALSE,
  seed = seed
)

completed <- lapply(seq_len(imputation_count), function(chain) {
  mice::complete(fit, action = chain)
})
metrics <- unlist(lapply(seq_along(completed), function(chain) {
  stats::setNames(
    vapply(completed[[chain]], mean, numeric(1L)),
    paste0("chain_", chain, "_mean_", names(completed[[chain]]))
  )
}), use.names = TRUE)

reference <- list(
  schema_version = 1L,
  artifact_kind = "evidence-only",
  reference_tool = "R mice",
  reference_tool_version = as.character(utils::packageVersion("mice")),
  r_version = as.character(getRversion()),
  platform = R.version$platform,
  method = "mice::mice(method='norm', m=3, maxit=5)",
  seed = seed,
  dataset_sha256 = Sys.getenv("DATASET_SHA256", unset = "unrecorded"),
  source_script_sha256 = Sys.getenv("REFERENCE_SCRIPT_SHA256", unset = "unrecorded"),
  metrics = as.list(metrics)
)

jsonlite::write_json(reference, output_path, auto_unbox = TRUE, digits = 17, pretty = TRUE)
