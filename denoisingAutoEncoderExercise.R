# Denoising Autoencoder
source('deeplearning/common.R')
source('deeplearning/softmax.R')
source('deeplearning/denoisingAutoencoder.R')

hiddenSize <- 500
batchSize <- 20
alpha <- 0.1

trainData <- loadImageFile('data/train-images-idx3-ubyte')

m <- ncol(trainData)
inputSize <- nrow(trainData)
numbatches <- m / batchSize

r <- 4 * sqrt(6) / sqrt(hiddenSize+inputSize)
W <- matrix(runif(hiddenSize * inputSize) * 2 * r - r, hiddenSize, inputSize)
b <- rep(0, inputSize)
c <- rep(0, hiddenSize)
theta <- c(W, b, c)

for(l in 1:15) {
  kk <- sample(1:m)
  cost <- 0
  for(i in 1:numbatches) {
    dat <- trainData[, kk[((i - 1)*batchSize+1) : (i*batchSize)]]
    dat <- rbinom(batchSize * inputSize, 1, 0.7) * dat

    cost <- cost + denoisingAutoencoderCost(theta, inputSize, hiddenSize, dat)
    theta <- theta -  alpha * denoisingAutoencoderGrad(theta, inputSize, hiddenSize, dat)
  }
  print(sprintf("At iterate %s = %s", l, cost / numbatches))
}
W = matrix(theta[1 : (hiddenSize*inputSize)], hiddenSize, inputSize)
displayNetwork(W[1:100, ])


trainData <- loadImageFile('data/train-images-idx3-ubyte')
trainLabels <- loadLabelFile('data/train-labels-idx1-ubyte')
numLabels <- length(table(trainLabels))
testData <- loadImageFile('data/t10k-images-idx3-ubyte')
testLabels <- loadLabelFile('data/t10k-labels-idx1-ubyte')

trainFeatures <- feedForwardAutoencoder(theta, hiddenSize, inputSize, trainData)
testFeatures <- feedForwardAutoencoder(theta, hiddenSize, inputSize, testData)

softmaxLambda <- 1e-4
softmaxTheta <- 0.005 * runif(numLabels * hiddenSize)
softmaxOptTheta <- optim(softmaxTheta,
                         function(theta) softmaxCost(theta, numLabels, hiddenSize, softmaxLambda, trainFeatures, trainLabels),
                         function(theta) softmaxGrad(theta, numLabels, hiddenSize, softmaxLambda, trainFeatures, trainLabels),
                         method = "L-BFGS-B", control = list(trace = 3, maxit = 500))$par

softmaxPredict(softmaxOptTheta, testFeatures, testLabels)
softmaxPredict(softmaxOptTheta, trainFeatures, trainLabels)


#**************************************************************************************************
r <- 4 * sqrt(6) / sqrt(hiddenSize+inputSize)
W <- matrix(runif(hiddenSize * inputSize) * 2 * r - r, hiddenSize, inputSize)
bhid <- rep(0, hiddenSize)
bvis <- rep(0, inputSize)

for(l in 1:10) {
  kk <- sample(1:m)
  cost <- 0
  for(i in 1:numbatches) {
    data <- trainData[, kk[((i - 1)*batchSize+1) : (i*batchSize)]]
    data <- rbinom(batchSize * inputSize, 1, 0.7) * data

    z2 <- W %*% data + bhid
    a2 <- sigmoid(z2)
    z3 <- t(W) %*% a2 + bvis
    a3 <- sigmoid(z3)

    LGrad <- data / a3 - (1 - data) / (1 - a3)
    a3Grad <- a3 * (1 - a3)
    a2Grad <- a2 * (1 - a2)
    a3GradLGrad <- a3Grad * LGrad
    Wa3GradLGrad <- W %*% a3GradLGrad

    WGrad <- -(a2Grad * Wa3GradLGrad) %*% t(data) - a2 %*% t(a3GradLGrad)
    W <- W - alpha * WGrad / batchSize

    bhidGrad <- -rowMeans(Wa3GradLGrad * a2Grad)
    bhid <- bhid - alpha * bhidGrad

    bvisGrad <- -rowMeans(a3GradLGrad)
    bvis <- bvis - alpha * bvisGrad

    L <- -colSums(data * log(a3) + (1 - data) * log(1 - a3))
    cost <- cost + mean(L)
  }
  print(sprintf("At iterate %s = %s", l, cost / numbatches))
}

trainData <- loadImageFile('data/train-images-idx3-ubyte')
trainLabels <- loadLabelFile('data/train-labels-idx1-ubyte')
numLabels <- length(table(trainLabels))
testData <- loadImageFile('data/t10k-images-idx3-ubyte')
testLabels <- loadLabelFile('data/t10k-labels-idx1-ubyte')

trainFeatures <- feedForwardAutoencoder(c(W, rep(0, hiddenSize)), hiddenSize, inputSize, trainData)
testFeatures <- feedForwardAutoencoder(c(W, rep(0, hiddenSize)), hiddenSize, inputSize, testData)

softmaxLambda <- 1e-4
softmaxTheta <- 0.005 * runif(numLabels * hiddenSize)
softmaxOptTheta <- optim(softmaxTheta,
                         function(theta) softmaxCost(theta, numLabels, hiddenSize, softmaxLambda, trainFeatures, trainLabels),
                         function(theta) softmaxGrad(theta, numLabels, hiddenSize, softmaxLambda, trainFeatures, trainLabels),
                         method = "L-BFGS-B", control = list(trace = 3, maxit = 500))$par

softmaxPredict(softmaxOptTheta, testFeatures, testLabels)
softmaxPredict(softmaxOptTheta, trainFeatures, trainLabels)

