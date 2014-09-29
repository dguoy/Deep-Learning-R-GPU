#cnn
library(R.matlab)
library(ggplot2)
library(reshape)

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

convImages <- stlTrainSubset$trainImages[, , , 1:8]

system.time(convolvedFeatures <- cnnConvolve(patchDim, hiddenSize, convImages, W, b, ZCAWhite, meanPatch))
for(i in 1:1000) {
	featureNum <- sample(1:hiddenSize, 1)
	imageNum <- sample(1:8, 1)
	imageRow <- sample(1:(imageDim - patchDim + 1), 1)
	imageCol <- sample(1:(imageDim - patchDim + 1), 1)
	
	patch <- as.vector(convImages[imageRow:(imageRow + patchDim - 1), imageCol:(imageCol + patchDim - 1), , imageNum])     
	patch <- patch - meanPatch
	patch <- ZCAWhite %*% patch
	
	features <- feedForwardAutoencoder(optTheta, hiddenSize, visibleSize, patch)
	
	if(abs(features[featureNum, 1] - convolvedFeatures[imageRow, imageCol, featureNum, imageNum]) > 1e-9) {
		print('error')
	}
}

testMatrix <- matrix(1:64, 8, 8)
expectedMatrix <- matrix(c(mean(testMatrix[1:4, 1:4]),
				mean(testMatrix[5:8, 1:4]),
				mean(testMatrix[1:4, 5:8]),
				mean(testMatrix[5:8, 5:8])), 2, 2)
testMatrix <- array(testMatrix, c(8, 8, 1, 1))
pooledFeatures <- cnnPool(4, testMatrix)

if(abs(sum(as.vector(pooledFeatures) - as.vector(expectedMatrix))) > 1e-9) {
	print('error')
}

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

#**********************************************************************************
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

