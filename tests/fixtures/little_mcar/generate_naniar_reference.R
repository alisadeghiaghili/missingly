#!/usr/bin/env Rscript

# Regenerate a machine-readable Little's MCAR reference from a locked R/naniar
# environment. This script writes scalar evidence only; it never serialises rows.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: generate_naniar_reference.R <airquality.csv> <output.json>")
}

input_path <- args[[1L]]
output_path <- args[[2L]]
if (!file.exists(input_path)) {
  stop("Input fixture does not exist: ", input_path)
}

for (package_name in c("jsonlite", "naniar")) {
  if (!requireNamespace(package_name, quietly = TRUE)) {
    stop("Required package is unavailable: ", package_name)
  }
}

fixture <- utils::read.csv(input_path, check.names = FALSE)
if ("rownames" %in% names(fixture)) {
  fixture[["rownames"]] <- NULL
}

result <- naniar::mcar_test(fixture)
reference <- list(
  schema_version = 1L,
  reference_tool = "R naniar",
  reference_tool_version = as.character(utils::packageVersion("naniar")),
  r_version = as.character(getRversion()),
  platform = R.version$platform,
  method = "naniar::mcar_test(airquality)",
  dataset_sha256 = Sys.getenv("DATASET_SHA256", unset = "unrecorded"),
  source_script_sha256 = Sys.getenv("REFERENCE_SCRIPT_SHA256", unset = "unrecorded"),
  metrics = list(
    chi_square = as.numeric(result[["statistic"]][[1L]]),
    degrees_of_freedom = as.numeric(result[["df"]][[1L]]),
    p_value = as.numeric(result[["p.value"]][[1L]]),
    missing_patterns = as.numeric(result[["missing.patterns"]][[1L]])
  )
)

jsonlite::write_json(reference, output_path, auto_unbox = TRUE, digits = 17, pretty = TRUE)
