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
		ab <- a1 %**% b1
		ab <- ab + a2 %**% b2
	} else if(ncol(b) > maxDim) {
		b1 <- b[,1:maxDim]
		b2 <- b[,(maxDim+1):ncol(b)]
		ab <- a %**% b1
		ab <- cbind(ab, a %**% b2)
	} else if(nrow(a) > maxDim) {
		a1 <- a[1:maxDim,]
		a2 <- a[(maxDim+1):nrow(b),]
		ab <- a1 %**% b
		ab <- rbind(ab, a2 %**% b)
	} else {
		ab <- .Call("gpuMatMult", a, b)
	}
	ab
}
