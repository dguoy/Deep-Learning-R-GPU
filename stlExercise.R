# Self-Taught Learning
source('deeplearning/common.R')
source('deeplearning/sparseAutoencoder.R')
source('deeplearning/softmax.R')

inputSize <- 28 * 28
numLabels <- 5
hiddenSize <- 200
sparsityParam <- 0.1
lambda <- 3e-3
beta <- 3
maxIter <- 500

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

testSet <- labeledSet[(numTrain+1) : length(labeledSet)]
testData <- mnistData[, testSet]
testLabels <- mnistLabels[testSet] + 1

stlTheta <- initializeParameters(hiddenSize, inputSize)
#************************************************ With method ***************************************************************************************

stlOptTheta <- optim(stlTheta,
						function(theta) sparseAutoencoderCost(theta, inputSize, hiddenSize, lambda, sparsityParam, beta, unlabeledData),
						function(theta) sparseAutoencoderGrad(theta, inputSize, hiddenSize, lambda, sparsityParam, beta, unlabeledData),
						method = "L-BFGS-B", control = list(trace = 3, maxit = maxIter))$par

trainFeatures <- feedForwardAutoencoder(stlOptTheta, hiddenSize, inputSize, trainData)
testFeatures <- feedForwardAutoencoder(stlOptTheta, hiddenSize, inputSize, testData)

softmaxLambda <- 1e-4
softmaxTheta <- 0.005 * runif(numLabels * hiddenSize)
softmaxOptTheta <- optim(softmaxTheta,
							function(theta) softmaxCost(theta, numLabels, hiddenSize, softmaxLambda, trainFeatures, trainLabels),
							function(theta) softmaxGrad(theta, numLabels, hiddenSize, softmaxLambda, trainFeatures, trainLabels),
							method = "L-BFGS-B", control = list(trace = 3, maxit = maxIter))$par

softmaxPredict(softmaxOptTheta, testFeatures, testLabels)
softmaxPredict(softmaxOptTheta, trainFeatures, trainLabels)

#************************************************ With Object-oriented ******************************************************************************
sparseAutoencoder <- SparseAutoencoder$new(inputSize, hiddenSize, lambda, sparsityParam, beta, unlabeledData)
stlOptTheta <- optim(stlTheta, sparseAutoencoder$cost, sparseAutoencoder$grad, method = "L-BFGS-B", control = list(trace = 3, maxit = maxIter))$par

trainFeatures <- feedForwardAutoencoder(stlOptTheta, hiddenSize, inputSize, trainData)
testFeatures <- feedForwardAutoencoder(stlOptTheta, hiddenSize, inputSize, testData)

softmaxLambda <- 1e-4
softmaxTheta <- 0.005 * runif(numLabels * hiddenSize)

softmax <- Softmax$new(numLabels, hiddenSize, softmaxLambda, trainFeatures, trainLabels)
softmaxOptTheta <- optim(softmaxTheta, softmax$cost, softmax$grad, method = "L-BFGS-B", control = list(trace = 3, maxit = maxIter))$par
softmax$predict(softmaxOptTheta, testFeatures, testLabels)
softmax$predict(softmaxOptTheta, trainFeatures, trainLabels)