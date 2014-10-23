source('common.R')
source('deepLearning.R')
source('softmax.R')

inputSize <- 28 * 28
numClasses <- 10

lambda <- 1e-4

trainData <- loadImageFile('data/train-images-idx3-ubyte')
trainLabels <- loadLabelFile('data/train-labels-idx1-ubyte')

theta <- 0.005 * runif(numClasses * inputSize)

optimTheta <- optim(theta,
		function(theta) softmaxCost(theta, numClasses, inputSize, lambda, trainData, trainLabels),
		function(theta) softmaxGrad(theta, numClasses, inputSize, lambda, trainData, trainLabels),
		method = "L-BFGS-B", control = list(trace = 3, maxit = 500))$par

testData <- loadImageFile('data/t10k-images-idx3-ubyte')
testLabels <- loadLabelFile('data/t10k-labels-idx1-ubyte')
softmaxPredict(optimTheta, testData, testLabels)

#************************************************ With Object-oriented ******************************************************************************
softmax <- Softmax$new(numClasses, inputSize, lambda, trainData, trainLabels)
optimTheta <- optim(theta, softmax$cost, softmax$grad, method = "L-BFGS-B", control = list(trace = 3, maxit = 500))$par
softmax$predict(optimTheta, testData, testLabels)
softmax$predict(optimTheta, trainData, trainLabels)
