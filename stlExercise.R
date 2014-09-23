inputSize <- 28 * 28
numLabels <- 5
hiddenSize <- 200
sparsityParam <- 0.1
lambda <- 3e-3
beta <- 3
maxIter <- 400

mnistData <- train$x
mnistLabels <- train$y
mnistLabels[mnistLabels == 10] <- 0

unlabeledSet <- mnistLabels >= 5
unlabeledData <- mnistData[, unlabeledSet]

labeledSet <- mnistLabels >= 0 & mnistLabels <= 4
numTrain <- round(length(labeledSet)/2)

trainSet <- labeledSet[1 : numTrain]
trainData <- mnistData[, trainSet]
trainLabels <- mnistLabels[trainSet] + 1

testSet <- labeledSet[numTrain+1 : length(labeledSet)]
testData <- mnistData[, testSet]
testLabels <- mnistLabels[testSet] + 1

theta <- initializeParameters(hiddenSize, inputSize)

optimTheta <- optim(theta,
		function(theta) J(theta, inputSize, hiddenSize, lambda, sparsityParam, beta, unlabeledData),
		function(theta) g(theta, inputSize, hiddenSize, lambda, sparsityParam, beta, unlabeledData),
		method = "L-BFGS-B", control = list(trace = 3, maxit = 400))$par

trainFeatures <- feedForwardAutoencoder(optimTheta, hiddenSize, inputSize, trainData)
testFeatures <- feedForwardAutoencoder(optimTheta, hiddenSize, inputSize, testData)

#*************************************************** Function **********************************************************************

fr <- function(theta) J(theta, inputSize, hiddenSize, lambda, sparsityParam, beta, unlabeledData)
grr <- function(theta) g(theta, inputSize, hiddenSize, lambda, sparsityParam, beta, unlabeledData)

feedForwardAutoencoder <- function(theta, hiddenSize, visibleSize, data) {
	W1 <- matrix(theta[1:(hiddenSize*visibleSize)], hiddenSize, visibleSize)
	b1 <- theta[(2*hiddenSize*visibleSize+1) : (2*hiddenSize*visibleSize+hiddenSize)]

	z2 <- W1 %*% data + b1
	a2 <- sigmoid(z2)
	a2
}
