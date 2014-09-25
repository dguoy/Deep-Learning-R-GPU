library(R.matlab)
library(reshape2)
library(ggplot2)

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

system.time(
	optTheta <- optim(theta,
		function(theta) sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches),
		function(theta) sparseAutoencoderLinearGrad(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches),
		method = "L-BFGS-B", control = list(trace = 3, maxit = 500))$par)

checkNumericalGradient(theta,
		function(theta) sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches),
		function(theta) sparseAutoencoderLinearGrad(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches))

#**********************************************************************************
sparseAutoencoderLinearCost <- function(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data) {
	W1 = matrix(theta[1 : (hiddenSize*visibleSize)], hiddenSize, visibleSize)
	W2 = matrix(theta[(hiddenSize*visibleSize+1) : (2*hiddenSize*visibleSize)], visibleSize, hiddenSize)
	b1 = theta[(2*hiddenSize*visibleSize+1) : (2*hiddenSize*visibleSize+hiddenSize)]
	b2 = theta[(2*hiddenSize*visibleSize+hiddenSize+1) : length(theta)]

	ndims <- nrow(data)
	m <- ncol(data)
	y <- data

	z2 <- W1 %*% data + b1
	a2 <- sigmoid(z2)
	z3 <- W2 %*% a2 + b2
	a3 <- z3

	rho <- rowMeans(a2)

	(1 /  (2 * m)) * sum((a3 - y)^2) +
		(lambda / 2) * (sum(W1^2) + sum(W2^2)) +
			beta * sum(sparsityParam * log(sparsityParam / rho) + (1 - sparsityParam) * log((1 - sparsityParam) / (1-rho)))
}
sparseAutoencoderLinearGrad <- function(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data) {
	W1 = matrix(theta[1 : (hiddenSize*visibleSize)], hiddenSize, visibleSize)
	W2 = matrix(theta[(hiddenSize*visibleSize+1) : (2*hiddenSize*visibleSize)], visibleSize, hiddenSize)
	b1 = theta[(2*hiddenSize*visibleSize+1) : (2*hiddenSize*visibleSize+hiddenSize)]
	b2 = theta[(2*hiddenSize*visibleSize+hiddenSize+1) : length(theta)]

	ndims <- nrow(data)
	m <- ncol(data)
	y <- data

	z2 <- W1 %*% data + b1
	a2 <- sigmoid(z2)
	z3 <- W2 %*% a2 + b2
	a3 <- z3

	rho <- rowMeans(a2)
	sparsity_delta <- -sparsityParam / rho + (1-sparsityParam) / (1-rho)

	delta3 <- -(y - a3)
	delta2 <- (t(W2) %*% delta3 + beta * sparsity_delta) * a2 * (1 - a2)

	deltaW1 <- delta2 %*% t(data)
	deltab1 <- rowSums(delta2)
	deltaW2 <- delta3 %*% t(a2)
	deltab2 <- rowSums(delta3)

	W1grad <- (1 / m) * deltaW1 + lambda * W1
	b1grad <- (1 / m) * deltab1
	W2grad <- (1 / m) * deltaW2 + lambda * W2
	b2grad <- (1 / m) * deltab2

	c(as.vector(W1grad), as.vector(W2grad), as.vector(b1grad), as.vector(b2grad))
}
