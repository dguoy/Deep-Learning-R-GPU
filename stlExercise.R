inputSize <- 28 * 28
numLabels <- 5
hiddenSize <- 200
sparsityParam <- 0.1
lambda <- 3e-3
beta <- 3
maxIter <- 400

mnistData <- loadImageFile('data/train-images-idx3-ubyte')
mnistLabels <- loadLabelFile('data/train-labels-idx1-ubyte')
mnistLabels[mnistLabels == 10] <- 0

unlabeledSet <- which(mnistLabels >= 5)
unlabeledData <- mnistData[, unlabeledSet]

labeledSet <- which(mnistLabels >= 0 & mnistLabels <= 4)
numTrain <- round(length(labeledSet)/2)

trainSet <- labeledSet[1 : numTrain]
trainData <- mnistData[, trainSet]
trainLabels <- mnistLabels[trainSet] + 1

testSet <- labeledSet[numTrain+1 : length(labeledSet)]
testData <- mnistData[, testSet]
testLabels <- mnistLabels[testSet] + 1

theta <- initializeParameters(hiddenSize, inputSize)

optimTheta <- optim(theta,
		function(theta) sparseAutoencoderCost(theta, inputSize, hiddenSize, lambda, sparsityParam, beta, unlabeledData),
		function(theta) sparseAutoencoderGrad(theta, inputSize, hiddenSize, lambda, sparsityParam, beta, unlabeledData),
		method = "L-BFGS-B", control = list(trace = 3, maxit = maxIter))$par

trainFeatures <- feedForwardAutoencoder(optimTheta, hiddenSize, inputSize, trainData)
testFeatures <- feedForwardAutoencoder(optimTheta, hiddenSize, inputSize, testData)
