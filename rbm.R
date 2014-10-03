library(R.matlab)
library(reshape2)
library(ggplot2)

hiddenSize <- 400
batchSize <- 100
alpha <- 1

x <- loadImageFile('data/train-images-idx3-ubyte')

m <- ncol(x)
inputSize <- nrow(x)
numbatches <- m / batchSize

r <- sqrt(6) / sqrt(hiddenSize+inputSize+1)
W <- matrix(runif(hiddenSize * inputSize) * 2 * r - r, ncol = inputSize, nrow = hiddenSize)
b <- rep(0, inputSize)
c <- rep(0, hiddenSize)

for(l in 1:10) {
	kk <- sample(1:m)
	err <- 0
	for(i in 1:numbatches) {
		v1 <- x[, kk[((i - 1)*batchSize+1) : (i*batchSize)]]
		h1 <- sigmoidRnd(W %*% v1 + c)
		v2 <- sigmoidRnd(t(W) %*% h1 + b)
		h2 <- sigmoidRnd(W %*% v2 + c)
		c1 <- h1 %*% t(v1)
		c2 <- h2 %*% t(v2)
	
		W <- W + alpha * (c1 - c2) / batchSize
		b <- b + alpha * rowMeans(v1 - v2)
		c <- c + alpha * rowMeans(h1 - h2)
		err <- err + sum((v1 - v2)^2) / batchSize
	}
	print(err)
}
displayNetwork(W)

numLabels <- 10
trainData <- loadImageFile('data/train-images-idx3-ubyte')
trainLabels <- loadLabelFile('data/train-labels-idx1-ubyte')
testData <- loadImageFile('data/t10k-images-idx3-ubyte')
testLabels <- loadLabelFile('data/t10k-labels-idx1-ubyte')

trainFeatures <- feedForwardAutoencoder(c(as.vector(W), c), hiddenSize, inputSize, trainData)
testFeatures <- feedForwardAutoencoder(c(as.vector(W), c), hiddenSize, inputSize, testData)

softmaxLambda <- 1e-4
softmaxTheta <- 0.005 * runif(numLabels * hiddenSize)
softmaxOptTheta <- optim(softmaxTheta,
                         function(theta) softmaxCost(theta, numLabels, hiddenSize, softmaxLambda, trainFeatures, trainLabels),
                         function(theta) softmaxGrad(theta, numLabels, hiddenSize, softmaxLambda, trainFeatures, trainLabels),
                         method = "L-BFGS-B", control = list(trace = 3, maxit = 500))$par

softmaxPredict(softmaxOptTheta, testFeatures, testLabels)
softmaxPredict(softmaxOptTheta, trainFeatures, trainLabels)

#********************************************* Function **********************************************
sigmoidRnd <- function(z) {
	ifelse(sigmoid(z) > matrix(runif(length(z)), nrow = nrow(z), ncol = ncol(z)), 1, 0)
}
