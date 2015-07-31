# Restricted Boltzmann Machines
source('deeplearning/common.R')
source('deeplearning/softmax.R')

hiddenSize <- 500
batchSize <- 100
alpha <- 0.1

trainData <- loadImageFile('data/train-images-idx3-ubyte')

m <- ncol(trainData)
inputSize <- nrow(trainData)
numbatches <- m / batchSize

r <- 4 * sqrt(6) / sqrt(hiddenSize+inputSize)
W <- matrix(runif(hiddenSize * inputSize) * 2 * r - r, ncol = inputSize, nrow = hiddenSize)
b <- rep(0, inputSize)
c <- rep(0, hiddenSize)

for(l in 1:15) {
  kk <- sample(1:m)
  err <- 0
  for(i in 1:numbatches) {
    v <- trainData[, kk[((i - 1)*batchSize+1) : (i*batchSize)]]
    h <- sigmoidRnd(W %*% v + c)

    gibbs_sample <- gibbs_hvh(h, k=1)
    v_sample <- gibbs_sample$v_sample
    h_sample <- gibbs_sample$h_sample

    c1 <- h %*% t(v)
    c2 <- h_sample %*% t(v_sample)

    W <- W + alpha * (c1 - c2) / batchSize
    b <- b + alpha * rowMeans(v - v_sample)
    c <- c + alpha * rowMeans(h - h_sample)
    err <- err + sum((v - v_sample)^2) / batchSize
  }
  print(sprintf("At iterate %s = %s", l, err / numbatches))
}
displayNetwork(W[1:100, ])

trainData <- loadImageFile('data/train-images-idx3-ubyte')
trainLabels <- loadLabelFile('data/train-labels-idx1-ubyte')
numLabels <- length(table(trainLabels))
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

#*********************************************************************
gibbs_hvh <- function(h, k=1) {
  v_sample <- sigmoidRnd(t(W) %*% h + b)
  h_sample <- sigmoidRnd(W %*% v_sample + c)
  if(k == 1) {
    return(list('v_sample'=v_sample, 'h_sample'=h_sample))
  } else {
    return(gibbs_hvh(h_sample, k - 1))
  }
}
