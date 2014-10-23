#Stacked Autoencoders
source('common.R')
source('deepLearning.R')
source('softmax.R')

inputSize <- 28 * 28
numClasses <- 10
hiddenSizeL1 <- 200
hiddenSizeL2 <- 200
sparsityParam <- 0.1
lambda <- 3e-3
beta <- 3

trainData <- loadImageFile('data/train-images-idx3-ubyte')
trainLabels <- loadLabelFile('data/train-labels-idx1-ubyte')

sae1Theta <- initializeParameters(hiddenSizeL1, inputSize)
sae1OptTheta <- optim(sae1Theta,
						function(theta) sparseAutoencoderCost(theta, inputSize, hiddenSizeL1, lambda, sparsityParam, beta, trainData),
						function(theta) sparseAutoencoderGrad(theta, inputSize, hiddenSizeL1, lambda, sparsityParam, beta, trainData),
						method = "L-BFGS-B", control = list(trace = 3, maxit = 400))$par
sae1Features <- feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, inputSize, trainData)

sae2Theta <- initializeParameters(hiddenSizeL2, hiddenSizeL1)
sae2OptTheta <- optim(sae2Theta,
						function(theta) sparseAutoencoderCost(theta, hiddenSizeL1, hiddenSizeL2, lambda, sparsityParam, beta, sae1Features),
						function(theta) sparseAutoencoderGrad(theta, hiddenSizeL1, hiddenSizeL2, lambda, sparsityParam, beta, sae1Features),
						method = "L-BFGS-B", control = list(trace = 3, maxit = 400))$par
sae2Features <- feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, hiddenSizeL1, sae1Features)

softmaxInputSize <- nrow(sae2Features)
numClasses <- length(table(trainLabels))
softmaxLambda <- 1e-4
softmaxTheta <- 0.005 * runif(hiddenSizeL2 * numClasses)
softmaxOptTheta <- optim(softmaxTheta,
							function(theta) softmaxCost(theta, numClasses, softmaxInputSize, softmaxLambda, sae2Features, trainLabels),
							function(theta) softmaxGrad(theta, numClasses, softmaxInputSize, softmaxLambda, sae2Features, trainLabels),
							method = "L-BFGS-B", control = list(trace = 3, maxit = 400))$par

stackedAETheta <- c(softmaxOptTheta,
sae1OptTheta[1:(hiddenSizeL1 * inputSize + hiddenSizeL1)],
sae2OptTheta[1:(hiddenSizeL2 * hiddenSizeL1 + hiddenSizeL2)])

stackedAEOptTheta <- optim(stackedAETheta,
							function(theta) stackedAECost(theta, inputSize, hiddenSizeL2, numClasses, lambda, trainData, trainLabels),
							function(theta) stackedAEGrad(theta, inputSize, hiddenSizeL2, numClasses, lambda, trainData, trainLabels),
							method = "L-BFGS-B", control = list(trace = 3, maxit = 400))$par

testData <- loadImageFile('data/t10k-images-idx3-ubyte')
testLabels <- loadLabelFile('data/t10k-labels-idx1-ubyte')

predict <- stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, numClasses, testData)
mean(predict == testLabels)
predict <- stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, numClasses, testData)
mean(predict == testLabels)
