imageChannels <- 3

patchDim <- 8
numPatches <- 10000

visibleSize <- patchDim * patchDim * imageChannels
outputSize <- visibleSize
hiddenSize <- 400

sparsityParam <- 0.035
lambda <- 3e-3
beta <- 5

epsilon <- 0.1

patches <- sampleColorImages(images, numpatches=numPatches)
meanPatch <- rowMeans(patches)
patches <- patches - meanPatch
sigma <- patches %*% t(patches) / numPatches
sigma.svd <- svd(sigma)
ZCAWhite <- sigma.svd$u %*% diag(1 / sqrt(sigma.svd$d + epsilon)) %*% t(sigma.svd$u)
patches <- ZCAWhite %*% patches

theta <- initializeParameters(hiddenSize, visibleSize)

optTheta <- optim(theta,
		function(theta) sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches),
		function(theta) sparseAutoencoderLinearGrad(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches),
		method = "L-BFGS-B", control = list(trace = 3, maxit = 500))$par


W <- matrix(optTheta[1:(visibleSize * hiddenSize)], hiddenSize, visibleSize)
b <- optTheta[(hiddenSize*visibleSize+1):(hiddenSize*visibleSize+hiddenSize)]

imageDim <- dim(images)[1]
numTrainImages <- dim(images)[4]
poolDim <- 30

convolvedFeatures <- cnnConvolve(patchDim, hiddenSize, images, W, b, ZCAWhite, meanPatch)
pooledFeaturesTrain <- cnnPool(poolDim, convolvedFeatures)
pooledFeaturesTrain <- matrix(pooledFeaturesTrain, ncol=numTrainImages)

dataLabels <- rep(1:2, each=40)
trainSet <- sample(1:80, 40, replace=F)
trainData <- pooledFeaturesTrain[, trainSet]
trainLabels <- dataLabels[trainSet]
testData <- pooledFeaturesTrain[, -trainSet]
testLabels <- dataLabels[-trainSet]

softmaxLambda <- 1e-4
softmaxTheta <- 0.005 * runif(2 * 3600)
softmaxOptTheta <- optim(softmaxTheta,
		function(theta) softmaxCost(theta, 2, 3600, softmaxLambda, trainData, trainLabels),
		function(theta) softmaxGrad(theta, 2, 3600, softmaxLambda, trainData, trainLabels),
		method = "L-BFGS-B", control = list(trace = 3, maxit = 500))$par

softmaxPredict(softmaxOptTheta, testData, testLabels)
softmaxPredict(softmaxOptTheta, trainData, trainLabels)

#*******************************************************************************************************************

library(png)

images <- array(0, c(97, 97, 3, length(imagePaths)))

for(i in 1:length(imagePaths)) {
	images[,,1,i] <- rotate(readPNG(paste('/Users/jp61130/git/ufldl/data/stickers', imagePaths[i], sep='/'))[,,1])
	images[,,2,i] <- rotate(readPNG(paste('/Users/jp61130/git/ufldl/data/stickers', imagePaths[i], sep='/'))[,,2])
	images[,,3,i] <- rotate(readPNG(paste('/Users/jp61130/git/ufldl/data/stickers', imagePaths[i], sep='/'))[,,3])
}

imagePaths <- c(
	'27320.png',
	'27321.png',
	'27322.png',
	'27323.png',
	'27324.png',
	'27325.png',
	'27326.png',
	'27327.png',
	'27328.png',
	'27329.png',
	'27330.png',
	'27331.png',
	'27332.png',
	'27333.png',
	'27334.png',
	'27335.png',
	'27336.png',
	'27337.png',
	'27338.png',
	'27339.png',
	'27340.png',
	'27341.png',
	'27342.png',
	'27343.png',
	'27344.png',
	'27345.png',
	'27346.png',
	'27347.png',
	'27348.png',
	'27349.png',
	'27350.png',
	'27351.png',
	'27352.png',
	'27353.png',
	'27354.png',
	'27355.png',
	'27356.png',
	'27357.png',
	'27358.png',
	'27359.png',
	'130857.png',
	'130858.png',
	'130859.png',
	'130860.png',
	'130861.png',
	'130862.png',
	'130863.png',
	'130864.png',
	'130865.png',
	'130866.png',
	'130867.png',
	'130868.png',
	'130869.png',
	'130870.png',
	'130871.png',
	'130872.png',
	'130873.png',
	'130874.png',
	'130875.png',
	'130876.png',
	'130877.png',
	'130878.png',
	'130879.png',
	'130880.png',
	'130881.png',
	'130882.png',
	'130883.png',
	'130884.png',
	'130885.png',
	'130886.png',
	'130887.png',
	'130888.png',
	'130889.png',
	'130890.png',
	'130891.png',
	'130892.png',
	'130893.png',
	'130894.png',
	'130895.png',
	'130896.png')
