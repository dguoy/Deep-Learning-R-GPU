library(R6)

#************************************************ Function *******************************************************************************************
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
#************************************************ With Object-oriented *******************************************************************************
StackedAE <- R6Class("StackedAE",
	private = list(
		inputSize=NA,
		hiddenSize=NA,
		numClasses=NA,
		lambda=NA,
		data=NA,
		labels=NA,
		a2=NA,
		a3=NA,
		numCases=NA,
		groundTruth=NA,
		M=NA,
		p=NA
	),
	public = list(
		initialize = function(inputSize, hiddenSize, numClasses, lambda, data, labels) {
			private$inputSize <- inputSize
			private$hiddenSize <- hiddenSize
			private$numClasses <- numClasses
			private$lambda <- lambda
			private$data <- data
			private$labels <- labels
		},
		cost = function(theta) {
			softmaxTheta = matrix(theta[1:(private$hiddenSize*private$numClasses)], private$numClasses, private$hiddenSize)
			W1 <- matrix(theta[(private$hiddenSize*private$numClasses+1):(private$hiddenSize*private$numClasses+private$hiddenSize*private$inputSize)], private$hiddenSize, private$inputSize)
			b1 <- theta[(private$hiddenSize*private$numClasses+private$hiddenSize*private$inputSize+1):(private$hiddenSize*private$numClasses+private$hiddenSize*private$inputSize+private$hiddenSize)]
			W2 <- matrix(theta[(private$hiddenSize*private$numClasses+private$hiddenSize*private$inputSize+private$hiddenSize+1):(private$hiddenSize*private$numClasses+private$hiddenSize*private$inputSize+private$hiddenSize+private$hiddenSize*private$hiddenSize)], private$hiddenSize, private$hiddenSize)
			b2 <- theta[(private$hiddenSize*private$numClasses+private$hiddenSize*private$inputSize+private$hiddenSize+private$hiddenSize*private$hiddenSize+1):(private$hiddenSize*private$numClasses+private$hiddenSize*private$inputSize+private$hiddenSize+private$hiddenSize*private$hiddenSize+private$hiddenSize)]
			
			z2 <- W1 %*% private$data + b1
			private$a2 <- sigmoid(z2)
			z3 <- W2 %*% private$a2 + b2
			private$a3 <- sigmoid(z3)
			
			private$numCases <- ncol(private$data)
			private$groundTruth <- matrix(0, private$numClasses, private$numCases)
			for(i in 1:length(private$labels)) {
				private$groundTruth[private$labels[i], i] <- 1
			}
			
			private$M <- softmaxTheta %*% private$a3
			private$p <- apply(private$M, 2, function(x) {y <- x - max(x); return(exp(y) / sum(exp(y)))})
			cost <- -(1 / private$numCases) * sum(private$groundTruth * log(private$p)) + (private$lambda / 2) * sum(softmaxTheta^2)
			cost
		},
		grad = function(theta) {
			softmaxTheta = matrix(theta[1:(private$hiddenSize*private$numClasses)], private$numClasses, private$hiddenSize)
			W1 <- matrix(theta[(private$hiddenSize*private$numClasses+1):(private$hiddenSize*private$numClasses+private$hiddenSize*private$inputSize)], private$hiddenSize, private$inputSize)
			b1 <- theta[(private$hiddenSize*private$numClasses+private$hiddenSize*private$inputSize+1):(private$hiddenSize*private$numClasses+private$hiddenSize*private$inputSize+private$hiddenSize)]
			W2 <- matrix(theta[(private$hiddenSize*private$numClasses+private$hiddenSize*private$inputSize+private$hiddenSize+1):(private$hiddenSize*private$numClasses+private$hiddenSize*private$inputSize+private$hiddenSize+private$hiddenSize*private$hiddenSize)], private$hiddenSize, private$hiddenSize)
			b2 <- theta[(private$hiddenSize*private$numClasses+private$hiddenSize*private$inputSize+private$hiddenSize+private$hiddenSize*private$hiddenSize+1):(private$hiddenSize*private$numClasses+private$hiddenSize*private$inputSize+private$hiddenSize+private$hiddenSize*private$hiddenSize+private$hiddenSize)]

			softmaxThetagrad <- -(1 / private$numCases) * (private$groundTruth - private$p) %*% t(private$a3) + private$lambda * softmaxTheta

			delta3 <- -(t(softmaxTheta) %*% (private$groundTruth - private$p)) * private$a3 * (1 - private$a3)
			delta2 <- t(W2) %*% delta3 * private$a2 * (1 - private$a2)
			
			gradW2 <- (1 / private$numCases) * delta3 %*% t(private$a2)
			gradb2 <- (1 / private$numCases) * rowSums(delta3)
			gradW1 <- (1 / private$numCases) * delta2 %*% t(private$data)
			gradb1 <- (1 / private$numCases) * rowSums(delta2)
			c(as.vector(softmaxThetagrad),
				as.vector(gradW1),
				as.vector(gradb1),
				as.vector(gradW2),
				as.vector(gradb2))
		},
		predict = function(theta, inputSize, hiddenSize, numClasses, data) {
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
	)
)
