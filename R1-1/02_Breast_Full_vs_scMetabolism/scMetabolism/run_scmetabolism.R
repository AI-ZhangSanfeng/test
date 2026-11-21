args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 5)
out <- args[[1]]; method <- args[[2]]; seed <- as.integer(args[[3]])
ncores <- as.integer(args[[4]]); expected_sha <- args[[5]]
required <- c("Matrix", "jsonlite", "scMetabolism", "VISION")
absent <- required[!vapply(required, requireNamespace, quietly=TRUE, FUN.VALUE=logical(1))]
if (length(absent)) stop(paste("Missing R packages:", paste(absent, collapse=", "),
  "Run install_scmetabolism.R using this Rscript first."))
desc <- utils::packageDescription("scMetabolism")
if (as.character(utils::packageVersion("scMetabolism")) != "0.2.2")
  stop("Please install the locked scMetabolism version 0.2.2.")
if (!is.null(desc$RemoteSha) && desc$RemoteSha != expected_sha)
  stop("scMetabolism RemoteSha differs from the locked official commit.")
RNGkind("L'Ecuyer-CMRG"); set.seed(seed)
counts <- methods::as(Matrix::readMM(file.path(out, "counts_genes_by_spots.mtx")), "CsparseMatrix")
genes <- readLines(file.path(out, "genes.txt"), encoding="UTF-8")
spots <- readLines(file.path(out, "spot_ids.txt"), encoding="UTF-8")
stopifnot(nrow(counts)==length(genes), ncol(counts)==length(spots),
          !anyDuplicated(genes), !anyDuplicated(spots))
rownames(counts) <- genes; colnames(counts) <- spots
if (any(!is.finite(counts@x)) || any(counts@x < 0) || any(Matrix::colSums(counts) <= 0)) stop("Invalid counts")
gmt <- system.file("data", "KEGG_metabolism_nc.gmt", package="scMetabolism")
if (!file.exists(gmt)) stop("The native KEGG GMT is missing from the installed package")
file.copy(gmt, file.path(out, "KEGG_metabolism_used.gmt"), overwrite=TRUE)
capture.output(scMetabolism::sc.metabolism, file=file.path(out,"scmetabolism_function_used.txt"))
result <- scMetabolism::sc.metabolism(countexp=counts, method=method,
                                    imputation=FALSE, ncores=ncores, metabolism.type="KEGG")
result <- as.matrix(result)
if (is.null(colnames(result)) || !setequal(colnames(result), spots))
  stop("Official scoring output IDs do not match the safe input identifiers")
result <- result[, spots, drop=FALSE]
utils::write.csv(result, file.path(out,"scores.csv"), row.names=TRUE, quote=TRUE)
utils::capture.output(sessionInfo(), file=file.path(out,"R_sessionInfo.txt"))
jsonlite::write_json(list(implementation="scMetabolism::sc.metabolism", method=method,
  imputation=FALSE, metabolism_type="KEGG", seed=seed, ncores=ncores,
  package_version=as.character(packageVersion("scMetabolism")),
  VISION_version=as.character(packageVersion("VISION")),
  installed_RemoteSha=desc$RemoteSha, requested_commit=expected_sha,
  package_path=find.package("scMetabolism"), Rscript=Sys.which("Rscript"),
  n_genes=nrow(counts), n_spots=ncol(counts), n_pathways=nrow(result)),
  file.path(out,"r_provenance.json"), auto_unbox=TRUE, pretty=TRUE)
