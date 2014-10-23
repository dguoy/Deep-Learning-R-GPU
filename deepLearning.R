sparseAutoencoderCost <- function(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data) {
	W1 = matrix(theta[1 : (hiddenSize*visibleSize)], hiddenSize, visibleSize)
	b1 = theta[(hiddenSize*visibleSize+1) : (hiddenSize*visibleSize+hiddenSize)]
	W2 = matrix(theta[(hiddenSize*visibleSize+hiddenSize+1) : (2*hiddenSize*visibleSize+hiddenSize)], visibleSize, hiddenSize)
	b2 = theta[(2*hiddenSize*visibleSize+hiddenSize+1) : length(theta)]
	
	m <- ncol(data)
	y <- data
	
	z2 <- W1 %*% data + b1
	a2 <- sigmoid(z2)
	z3 <- W2 %*% a2 + b2
	a3 <- sigmoid(z3)
	
	rho <- rowSums(a2) / m
	
	(1 /  (2 * m)) * sum((a3 - y)^2) +
		(lambda / 2) * (sum(W1^2) + sum(W2^2)) +
			beta * sum(sparsityParam * log(sparsityParam / rho) + (1 - sparsityParam) * log((1 - sparsityParam) / (1-rho)))
}
sparseAutoencoderGrad <- function(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data) {
	W1 = matrix(theta[1 : (hiddenSize*visibleSize)], hiddenSize, visibleSize)
	b1 = theta[(hiddenSize*visibleSize+1) : (hiddenSize*visibleSize+hiddenSize)]
	W2 = matrix(theta[(hiddenSize*visibleSize+hiddenSize+1) : (2*hiddenSize*visibleSize+hiddenSize)], visibleSize, hiddenSize)
	b2 = theta[(2*hiddenSize*visibleSize+hiddenSize+1) : length(theta)]
	
	m <- ncol(data)
	y <- data
	
	z2 <- W1 %*% data + b1
	a2 <- sigmoid(z2)
	z3 <- W2 %*% a2 + b2
	a3 <- sigmoid(z3)
	
	rho <- rowSums(a2) / m
	sparsity_delta <- -sparsityParam / rho + (1-sparsityParam) / (1-rho)
	
	delta3 <- -(y - a3) * a3 * (1 - a3)
	delta2 <- (t(W2) %*% delta3 + beta * sparsity_delta) * a2 * (1 - a2)
	
	deltaW1 <- delta2 %*% t(data)
	deltab1 <- rowSums(delta2)
	deltaW2 <- delta3 %*% t(a2)
	deltab2 <- rowSums(delta3)
	
	W1grad <- (1 / m) * deltaW1 + lambda * W1
	b1grad <- (1 / m) * deltab1
	W2grad <- (1 / m) * deltaW2 + lambda * W2
	b2grad <- (1 / m) * deltab2
	
	c(as.vector(W1grad), as.vector(b1grad), as.vector(W2grad), as.vector(b2grad))
}
#*****************************************************************************************************************************************************
sparseAutoencoderLinearCost <- function(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data) {
	W1 = matrix(theta[1 : (hiddenSize*visibleSize)], hiddenSize, visibleSize)
	b1 = theta[(hiddenSize*visibleSize+1) : (hiddenSize*visibleSize+hiddenSize)]
	W2 = matrix(theta[(hiddenSize*visibleSize+hiddenSize+1) : (2*hiddenSize*visibleSize+hiddenSize)], visibleSize, hiddenSize)
	b2 = theta[(2*hiddenSize*visibleSize+hiddenSize+1) : length(theta)]
	
	ndims <- nrow(data)
	m <- ncol(data)
	y <- data
	
	z2 <- W1 %*% data + b1
	a2 <- sigmoid(z2)
	z3 <- W2 %*% a2 + b2
	a3 <- z3
	
	rho <- rowMeans(a2)
	
	(1 /  (2 * m)) * sum((a3 - y)^2) +
		(lambda / 2) * (sum(W1^2) + sum(W2^2)) +
			beta * sum(sparsityParam * log(sparsityParam / rho) + (1 - sparsityParam) * log((1 - sparsityParam) / (1-rho)))
}
sparseAutoencoderLinearGrad <- function(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data) {
	W1 = matrix(theta[1 : (hiddenSize*visibleSize)], hiddenSize, visibleSize)
	b1 = theta[(hiddenSize*visibleSize+1) : (hiddenSize*visibleSize+hiddenSize)]
	W2 = matrix(theta[(hiddenSize*visibleSize+hiddenSize+1) : (2*hiddenSize*visibleSize+hiddenSize)], visibleSize, hiddenSize)
	b2 = theta[(2*hiddenSize*visibleSize+hiddenSize+1) : length(theta)]
	
	ndims <- nrow(data)
	m <- ncol(data)
	y <- data
	
	z2 <- W1 %*% data + b1
	a2 <- sigmoid(z2)
	z3 <- W2 %*% a2 + b2
	a3 <- z3
	
	rho <- rowMeans(a2)
	sparsity_delta <- -sparsityParam / rho + (1-sparsityParam) / (1-rho)
	
	delta3 <- -(y - a3)
	delta2 <- (t(W2) %*% delta3 + beta * sparsity_delta) * a2 * (1 - a2)
	
	deltaW1 <- delta2 %*% t(data)
	deltab1 <- rowSums(delta2)
	deltaW2 <- delta3 %*% t(a2)
	deltab2 <- rowSums(delta3)
	
	W1grad <- (1 / m) * deltaW1 + lambda * W1
	b1grad <- (1 / m) * deltab1
	W2grad <- (1 / m) * deltaW2 + lambda * W2
	b2grad <- (1 / m) * deltab2
	
	c(as.vector(W1grad), as.vector(b1grad), as.vector(W2grad), as.vector(b2grad))
}

#*****************************************************************************************************************************************************
stackedAECost <- function(theta, inputSize, hiddenSize, numClasses, lambda, data, labels) {
	softmaxTheta = matrix(theta[1:(hiddenSize*numClasses)], numClasses, hiddenSize)
	W1 <- matrix(theta[(hiddenSize*numClasses+1):(hiddenSize*numClasses+hiddenSize*inputSize)], hiddenSize, inputSize)
	b1 <- theta[(hiddenSize*numClasses+hiddenSize*inputSize+1):(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize)]
	W2 <- matrix(theta[(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+1):(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+hiddenSize*hiddenSize)], hiddenSize, hiddenSize)
	b2 <- theta[(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+hiddenSize*hiddenSize+1):(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+hiddenSize*hiddenSize+hiddenSize)]
	
	z2 <- W1 %*% data + b1
	a2 <- sigmoid(z2)
	z3 <- W2 %*% a2 + b2
	a3 <- sigmoid(z3)
	
	numCases <- ncol(data)
	groundTruth <- matrix(0, numClasses, numCases)
	for(i in 1:length(labels)) {
		groundTruth[labels[i], i] <- 1
	}
	
	M <- softmaxTheta %*% a3
	p <- apply(M, 2, function(x) {y <- x - max(x); return(exp(y) / sum(exp(y)))})
	cost <- -(1 / numCases) * sum(groundTruth * log(p)) + (lambda / 2) * sum(softmaxTheta^2)
	cost
}
stackedAEGrad <- function(theta, inputSize, hiddenSize, numClasses, lambda, data, labels) {
	softmaxTheta = matrix(theta[1:(hiddenSize*numClasses)], numClasses, hiddenSize)
	W1 <- matrix(theta[(hiddenSize*numClasses+1):(hiddenSize*numClasses+hiddenSize*inputSize)], hiddenSize, inputSize)
	b1 <- theta[(hiddenSize*numClasses+hiddenSize*inputSize+1):(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize)]
	W2 <- matrix(theta[(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+1):(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+hiddenSize*hiddenSize)], hiddenSize, hiddenSize)
	b2 <- theta[(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+hiddenSize*hiddenSize+1):(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+hiddenSize*hiddenSize+hiddenSize)]
	
	z2 <- W1 %*% data + b1
	a2 <- sigmoid(z2)
	z3 <- W2 %*% a2 + b2
	a3 <- sigmoid(z3)
	
	numCases <- ncol(data)
	groundTruth <- matrix(0, numClasses, numCases)
	for(i in 1:length(labels)) {
		groundTruth[labels[i], i] <- 1
	}
	
	M <- softmaxTheta %*% a3
	p <- apply(M, 2, function(x) {y <- x - max(x); return(exp(y) / sum(exp(y)))})
	softmaxThetagrad <- -(1 / numCases) * (groundTruth - p) %*% t(a3) + lambda * softmaxTheta
	
	delta3 <- -(t(softmaxTheta) %*% (groundTruth - p)) * a3 * (1 - a3)
	delta2 <- t(W2) %*% delta3 * a2 * (1 - a2)
	
	gradW2 <- (1 / numCases) * delta3 %*% t(a2)
	gradb2 <- (1 / numCases) * rowSums(delta3)
	gradW1 <- (1 / numCases) * delta2 %*% t(data)
	gradb1 <- (1 / numCases) * rowSums(delta2)
	c(as.vector(softmaxThetagrad),
			as.vector(gradW1),
			as.vector(gradb1),
			as.vector(gradW2),
			as.vector(gradb2))
}

stackedAEPredict <- function(theta, inputSize, hiddenSize, numClasses, data) {
	softmaxTheta = matrix(theta[1:(hiddenSize*numClasses)], numClasses, hiddenSize)
	W1 <- matrix(theta[(hiddenSize*numClasses+1):(hiddenSize*numClasses+hiddenSize*inputSize)], hiddenSize, inputSize)
	b1 <- theta[(hiddenSize*numClasses+hiddenSize*inputSize+1):(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize)]
	W2 <- matrix(theta[(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+1):(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+hiddenSize*hiddenSize)], hiddenSize, hiddenSize)
	b2 <- theta[(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+hiddenSize*hiddenSize+1):(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+hiddenSize*hiddenSize+hiddenSize)]
	
	z2 <- W1 %*% data + b1
	a2 <- sigmoid(z2)
	z3 <- W2 %*% a2 + b2
	a3 <- sigmoid(z3)
	
	predict <- apply(softmaxTheta %*% a3, 2, function(x) which.max(x))
	predict
}
#*****************************************************************************************************************************************************
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
#*****************************************************************************************************************************************************
