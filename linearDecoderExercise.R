# Linear Decoders with Autoencoders
source('deeplearning/common.R')
source('deeplearning/sparseAutoencoderLinear.R')

maxIter <- 500
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

patches <- readMat("data/stlSampledPatches.mat")[[1]]
meanPatch <- rowMeans(patches)
patches <- patches - meanPatch
sigma <- patches %*% t(patches) / numPatches
sigma.svd <- svd(sigma)
ZCAWhite <- sigma.svd$u %*% diag(1 / sqrt(sigma.svd$d + epsilon)) %*% t(sigma.svd$u)
patches <- ZCAWhite %*% patches

theta <- initializeParameters(hiddenSize, visibleSize)
#************************************************ With method ***************************************************************************************
optTheta <- optim(theta,
					function(theta) sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches),
					function(theta) sparseAutoencoderLinearGrad(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches),
					method = "L-BFGS-B", control = list(trace = 3, maxit = maxIter))$par

checkNumericalGradient(theta,
						function(theta) sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches),
						function(theta) sparseAutoencoderLinearGrad(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches))
#************************************************ With Object-oriented ******************************************************************************
sparseAutoencoderLinear <- SparseAutoencoderLinear$new(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches)
optimTheta <- optim(theta, sparseAutoencoderLinear$cost, sparseAutoencoderLinear$grad, method = "L-BFGS-B", control = list(trace = 3, maxit = maxIter))$par
