# Convolution and Pooling
source('deeplearning/common.R')
source('deepLearning.R')
source('softmax.R')

imageChannels <- 3

patchDim <- 8
numPatches <- 100000

visibleSize <- patchDim * patchDim * imageChannels
outputSize <- visibleSize
hiddenSize <- 400

sparsityParam <- 0.035
lambda <- 3e-3
beta <- 5

epsilon <- 0.1

poolDim <- 19

stlTrainSubset <- readMat("data/stlTrainSubset.mat")
imageDim <- dim(stlTrainSubset$trainImages)[1]

W <- matrix(optTheta[1:(visibleSize * hiddenSize)], hiddenSize, visibleSize)
b <- optTheta[(hiddenSize*visibleSize+1):(hiddenSize*visibleSize+hiddenSize)]

#softmax
stlTrainSubset <- readMat("data/stlTrainSubset.mat")
stlTestSubset <- readMat("data/stlTestSubset.mat")

pooledFeaturesTrain <- array(0, c((imageDim - patchDim + 1) / poolDim, (imageDim - patchDim + 1) / poolDim, hiddenSize, stlTrainSubset$numTrainImages))
for(i in 1:(stlTrainSubset$numTrainImages / 25)) {
	print(i)
	convolvedFeatures <- cnnConvolve(patchDim, hiddenSize, stlTrainSubset$trainImages[, , , ((i-1)*25+1):(i*25)], W, b, ZCAWhite, meanPatch)
	pooledFeaturesTrain[, , , ((i-1)*25+1):(i*25)] <- cnnPool(poolDim, convolvedFeatures)
}
pooledFeaturesTrain <- matrix(pooledFeaturesTrain, ncol=stlTrainSubset$numTrainImages)

pooledFeaturesTest <- array(0, c((imageDim - patchDim + 1) / poolDim, (imageDim - patchDim + 1) / poolDim, hiddenSize, stlTestSubset$numTestImages))
for(i in 1:(stlTestSubset$numTestImages / 25)) {
	print(i)
	convolvedFeatures <- cnnConvolve(patchDim, hiddenSize, stlTestSubset$testImages[, , , ((i-1)*25+1):(i*25)], W, b, ZCAWhite, meanPatch)
	pooledFeaturesTest[, , , ((i-1)*25+1):(i*25)] <- cnnPool(poolDim, convolvedFeatures)
}
pooledFeaturesTest <- matrix(pooledFeaturesTest, ncol=stlTestSubset$numTestImages)


softmaxInputSize <- nrow(pooledFeaturesTrain)
numClasses <- 4

softmaxLambda <- 1e-4

softmaxTheta <- 0.005 * runif(numClasses * softmaxInputSize)

softmaxOptTheta <- optim(softmaxTheta,
		function(theta) softmaxCost(theta, numClasses, softmaxInputSize, softmaxLambda, pooledFeaturesTrain, stlTrainSubset$trainLabels),
		function(theta) softmaxGrad(theta, numClasses, softmaxInputSize, softmaxLambda, pooledFeaturesTrain, stlTrainSubset$trainLabels),
		method = "L-BFGS-B", control = list(trace = 3, maxit = 500))$par
softmaxPredict(softmaxOptTheta, pooledFeaturesTest, stlTestSubset$testLabels)
softmaxPredict(softmaxOptTheta, pooledFeaturesTrain, stlTrainSubset$trainLabels)
