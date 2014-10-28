# Sparse Autoencoder
source('deeplearning/common.R')
source('deeplearning/sparseAutoencoder.R')

images <- readMat("data/IMAGES.mat")[[1]]

visibleSize <- 8*8
hiddenSize <- 25
lambda <- 0.0001
sparsityParam <- 0.01
beta <- 3

patches <- sampleImages(images)
theta <- initializeParameters(hiddenSize, visibleSize)
#************************************************ With method ***************************************************************************************

optimTheta <- optim(theta,
		function(theta) sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches),
		function(theta) sparseAutoencoderGrad(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches),
		method = "L-BFGS-B", control = list(trace = 3, maxit = 10))$par

W <- matrix(optimTheta[1 : (hiddenSize*visibleSize)], hiddenSize, visibleSize)
displayNetwork(W)
#************************************************ With Object-oriented ******************************************************************************
sparseAutoencoder <- SparseAutoencoder$new(visibleSize, hiddenSize, lambda, sparsityParam, beta, patches)
optimTheta <- optim(theta, sparseAutoencoder$cost, sparseAutoencoder$grad, method = "L-BFGS-B", control = list(trace = 3, maxit = 2000))$par

W <- matrix(optimTheta[1 : (hiddenSize*visibleSize)], hiddenSize, visibleSize)
displayNetwork(W)
