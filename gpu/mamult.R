dyn.load("gpu/matmult.so")
"%**%" <- function(a, b, maxDim = 50000) {
	a <- as.matrix(a)
	b <- as.matrix(b)

	if (ncol(a) != nrow(b))
		stop("error:  matrix dimensions mismatched for matrix multiplication")

	if(ncol(a) > maxDim) {
		a1 <- a[,1:maxDim]
		a2 <- a[,(maxDim+1):ncol(a)]
		b1 <- b[1:maxDim,]
		b2 <- b[(maxDim+1):nrow(b),]
		ab <- .Call("gpuMatMult", a1, b1)
		ab <- ab + .Call("gpuMatMult", a2, b2)
	} else if(ncol(b) > maxDim) {
		b1 <- b[,1:maxDim]
		b2 <- b[,(maxDim+1):ncol(b)]
		ab <- .Call("gpuMatMult", a, b1)
		ab <- cbind(ab, .Call("gpuMatMult", a, b2))
	} else {
		ab <- .Call("gpuMatMult", a, b)
	}
	ab
}
