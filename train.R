library(R.matlab)
library(reshape2)
library(ggplot2)

images <- readMat("data/IMAGES.mat")[[1]]

visibleSize <- 8*8
hiddenSize <- 25
lambda <- 0.0001
sparsityParam <- 0.01
beta <- 3

patches <- sampleImages(images)
theta <- initializeParameters(hiddenSize, visibleSize)

optimTheta <- optim(theta,
		function(theta) sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches),
		function(theta) sparseAutoencoderGrad(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches),
		method = "L-BFGS-B", control = list(trace = 3, maxit = 2000))$par

W <- matrix(optimTheta[1 : (hiddenSize*visibleSize)], hiddenSize, visibleSize)
displayNetwork(W)

#******************************************* function *********************************************************
sparseAutoencoderCost <- function(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data) {
	W1 = matrix(theta[1 : (hiddenSize*visibleSize)], hiddenSize, visibleSize)
	b1 = theta[(hiddenSize*visibleSize+1) : (hiddenSize*visibleSize+hiddenSize)]
	W2 = matrix(theta[(hiddenSize*visibleSize+hiddenSize+1) : (2*hiddenSize*visibleSize+hiddenSize)], visibleSize, hiddenSize)
	b2 = theta[(2*hiddenSize*visibleSize+hiddenSize+1) : length(theta)]

	m <- ncol(data)
	y <- data

	z2 <- W1 %*% data + b1
	a2 <- sigmoid(z2)
	z3 <- W2 %*% a2 + b2
	a3 <- sigmoid(z3)

	rho <- rowSums(a2) / m
	
	(1 /  (2 * m)) * sum((a3 - y)^2) +
		(lambda / 2) * (sum(W1^2) + sum(W2^2)) +
			beta * sum(sparsityParam * log(sparsityParam / rho) + (1 - sparsityParam) * log((1 - sparsityParam) / (1-rho)))
}
sparseAutoencoderGrad <- function(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data) {
  W1 = matrix(theta[1 : (hiddenSize*visibleSize)], hiddenSize, visibleSize)
  b1 = theta[(hiddenSize*visibleSize+1) : (hiddenSize*visibleSize+hiddenSize)]
  W2 = matrix(theta[(hiddenSize*visibleSize+hiddenSize+1) : (2*hiddenSize*visibleSize+hiddenSize)], visibleSize, hiddenSize)
  b2 = theta[(2*hiddenSize*visibleSize+hiddenSize+1) : length(theta)]

	m <- ncol(data)
	y <- data

	z2 <- W1 %*% data + b1
	a2 <- sigmoid(z2)
	z3 <- W2 %*% a2 + b2
	a3 <- sigmoid(z3)

	rho <- rowSums(a2) / m
	sparsity_delta <- -sparsityParam / rho + (1-sparsityParam) / (1-rho)

	delta3 <- -(y - a3) * a3 * (1 - a3)
	delta2 <- (t(W2) %*% delta3 + beta * sparsity_delta) * a2 * (1 - a2)

	deltaW1 <- delta2 %*% t(data)
	deltab1 <- rowSums(delta2)
	deltaW2 <- delta3 %*% t(a2)
	deltab2 <- rowSums(delta3)

	W1grad <- (1 / m) * deltaW1 + lambda * W1
	b1grad <- (1 / m) * deltab1
	W2grad <- (1 / m) * deltaW2 + lambda * W2
	b2grad <- (1 / m) * deltab2

	c(as.vector(W1grad), as.vector(b1grad), as.vector(W2grad), as.vector(b2grad))
}

