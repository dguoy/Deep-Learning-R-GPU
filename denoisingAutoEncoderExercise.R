# Sparse Autoencoder
source('deeplearning/common.R')
source('deeplearning/sparseAutoencoder.R')

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
denoisingAutoencoderCost <- function(theta, visibleSize, hiddenSize, data) {
  W = matrix(theta[1 : (hiddenSize*visibleSize)], hiddenSize, visibleSize)
  bhid = theta[(hiddenSize*visibleSize+1) : (hiddenSize*visibleSize+hiddenSize)]
  bvis = theta[(hiddenSize*visibleSize+hiddenSize+1) : length(theta)]

  z2 <- W %*% data + bhid
  a2 <- sigmoid(z2)
  z3 <- t(W) %*% a2 + bvis
  a3 <- sigmoid(z3)

  L <- -colSums(data * log(a3) + (1 - data) * log(1 - a3))
  return (mean(L))
}

denoisingAutoencoderGrad <- function(theta, visibleSize, hiddenSize, data) {
  W = matrix(theta[1 : (hiddenSize*visibleSize)], hiddenSize, visibleSize)
  bhid = theta[(hiddenSize*visibleSize+1) : (hiddenSize*visibleSize+hiddenSize)]
  bvis = theta[(hiddenSize*visibleSize+hiddenSize+1) : length(theta)]

  z2 <- W %*% data + bhid
  a2 <- sigmoid(z2)
  z3 <- t(W) %*% a2 + bvis
  a3 <- sigmoid(z3)

  WGrad <- -(((a2 * (1 - a2)) %*% t(data)) * as.vector(W %*% ((data / a3 - (1 - data) / (1 - a3)) * (a3 * (1 - a3)))) +
                t(t(a2 %*% t(a3 * (1 - a3))) * as.vector(data / a3 - (1 - data) / (1 - a3))))
  bhidGrad <- -as.vector(W %*% ((data / a3 - (1 - data) / (1 - a3)) * (a3 * (1 - a3))) * (a2 * (1 - a2)))
  bvisGrad <- -as.vector((data / a3 - (1 - data) / (1 - a3)) * (a3 * (1 - a3)))

  return (c(WGrad, bhidGrad, bvisGrad))
}

#************************************************ With method ***************************************************************************************
optimTheta <- optim(theta,
		function(theta) denoisingAutoencoderCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches),
		function(theta) denoisingAutoencoderGrad(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches),
		method = "L-BFGS-B", control = list(trace = 3, maxit = maxIter))$par

W <- matrix(optimTheta[1 : (hiddenSize*visibleSize)], hiddenSize, visibleSize)
displayNetwork(W)

#************************************************ With method ***************************************************************************************
i <- 25
j <- 64
-(sum((x / a3 - (1 - x) / (1 - a3)) * (a3 * (1 - a3)) * W[i, ] * (a2[i] * (1 - a2[i])) * x[j]) +
  (x[j] / a3[j] - (1 - x[j]) / (1 - a3[j])) * (a3[j] * (1 - a3[j])) * a2[i])


grad_W <- -(((a2 * (1 - a2)) %*% t(x)) * as.vector(W %*% ((x / a3 - (1 - x) / (1 - a3)) * (a3 * (1 - a3)))) +
  t(t(a2 %*% t(a3 * (1 - a3))) * as.vector(x / a3 - (1 - x) / (1 - a3))))

grad_bhid <- -as.vector(W %*% ((x / a3 - (1 - x) / (1 - a3)) * (a3 * (1 - a3))) * (a2 * (1 - a2)))

grad_bvis <- -as.vector((x / a3 - (1 - x) / (1 - a3)) * (a3 * (1 - a3)))

delta_theta <- rep(0, length(theta))
delta_theta[1689 - 66] <- epsilon
(denoisingAutoencoderCost(theta + delta_theta, visibleSize, hiddenSize, data) - denoisingAutoencoderCost(theta - delta_theta, visibleSize, hiddenSize, data)) / (2 * epsilon)


