
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
