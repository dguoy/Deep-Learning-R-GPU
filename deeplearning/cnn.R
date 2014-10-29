library(R6)

#************************************************ Function *******************************************************************************************
cnnConvolve <- function(patchDim, numFeatures, images, W, b, ZCAWhite, meanPatch) {
	numImages <- dim(images)[4]
	imageDim <- dim(images)[1]
	imageChannels <- dim(images)[3]
	convolvedDim <- imageDim - patchDim + 1
	
	convolvedFeatures <- array(0, c(convolvedDim, convolvedDim, numFeatures, numImages))
	
	WT <- W %*% ZCAWhite
	add <- b - WT %*% meanPatch
	
	for(imageNum in 1:numImages) {
		for(r in 1:convolvedDim) {
			for(c in 1:convolvedDim) {
				convolvedFeatures[r, c, , imageNum] <-
						sigmoid(WT %*% as.vector(images[r:(r+patchDim-1), c:(c+patchDim-1), ,imageNum]) + add)
			}
		}
	}
	convolvedFeatures
}
cnnPool <- function(poolDim, convolvedFeatures) {
	numImages <- dim(convolvedFeatures)[4]
	numFeatures <- dim(convolvedFeatures)[3]
	convolvedDim <- dim(convolvedFeatures)[1]
	poolLen <- convolvedDim / poolDim
	
	pooledFeatures <- array(0, c(poolLen, poolLen, numFeatures, numImages))
	for(i in 1 : numImages) {
		for(r in 1 : poolLen) {
			for(c in 1 : poolLen) {
				rb <- 1 + poolDim * (r-1)
				re <- poolDim * r
				cb <- 1 + poolDim * (c-1)
				ce <- poolDim * c
				pooledFeatures[r, c, , i] <-
						colMeans(matrix(convolvedFeatures[rb:re, cb:ce, ,i], convolvedDim * convolvedDim, numFeatures))
			}
		}
	}
	pooledFeatures
}
