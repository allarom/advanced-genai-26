#!/usr/bin/env Rscript

reports <- c("report.md", "baseline_repro_report.md")
output_dir <- "reports"

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

for (report in reports) {
  rmarkdown::render(
    input = report,
    output_format = "pdf_document",
    output_dir = output_dir,
    clean = TRUE,
    quiet = FALSE
  )
}
