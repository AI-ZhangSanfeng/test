# Use the dedicated TuMeNiche_scMetabolism R environment, not the NicheCompass Python environment.
options(repos=c(CRAN="https://cloud.r-project.org"), timeout=1200)
if (!requireNamespace("remotes", quietly=TRUE)) install.packages("remotes")
if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
cran <- c("data.table", "wesanderson", "Seurat", "SeuratObject", "ggplot2", "rsvd", "Matrix", "jsonlite")
need <- cran[!vapply(cran, requireNamespace, quietly=TRUE, FUN.VALUE=logical(1))]
if (length(need)) install.packages(need)
bioc <- c("AUCell", "GSEABase", "GSVA")
need <- bioc[!vapply(bioc, requireNamespace, quietly=TRUE, FUN.VALUE=logical(1))]
if (length(need)) BiocManager::install(need, ask=FALSE, update=FALSE)
if (!requireNamespace("VISION", quietly=TRUE))
  remotes::install_github("YosefLab/VISION@v3.0.2", upgrade="never")
remotes::install_github("wu-yc/scMetabolism@59d6d2b35482013a0ad8bc4db9d588aa8bb37b75",
                        upgrade="never", force=TRUE)
stopifnot(requireNamespace("scMetabolism", quietly=TRUE), requireNamespace("VISION", quietly=TRUE))
print(sessionInfo())
