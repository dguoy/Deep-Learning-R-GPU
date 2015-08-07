# Denoising Autoencoder
source('deeplearning/common.R')
source('deeplearning/denoisingAutoencoder.R')

images <- readMat("data/IMAGES.mat")[[1]]

maxIter <- 2000
visibleSize <- 8*8
hiddenSize <- 25

patches <- sampleImages(images)

r <- 4 * sqrt(6) / sqrt(hiddenSize+visibleSize)
W <- runif(hiddenSize * visibleSize, min = -r, max = r)
bhid <- rep(0, hiddenSize)
bvis <- rep(0, visibleSize)
theta <- c(W, bhid, bvis)

#************************************************ With method ***************************************************************************************
optimTheta <- optim(theta,
		function(theta) denoisingAutoencoderCost(theta, visibleSize, hiddenSize, patches),
		function(theta) denoisingAutoencoderGrad(theta, visibleSize, hiddenSize, patches),
		method = "L-BFGS-B", control = list(trace = 3, maxit = maxIter))$par

W <- matrix(optimTheta[1 : (hiddenSize*visibleSize)], hiddenSize, visibleSize)
displayNetwork(W)

