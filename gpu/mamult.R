dyn.load("gpu/matmult.so")
"%**%" <- function(a, b) {
	a <- as.matrix(a)
	b <- as.matrix(b)

	if (ncol(a) != nrow(b))
		stop("error:  matrix dimensions mismatched for matrix multiplication")

	.Call("gpuMatMult", a, b)
}
