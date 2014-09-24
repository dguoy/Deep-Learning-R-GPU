
inputSize <- 28 * 28
numClasses <- 10
hiddenSizeL1 <- 200
hiddenSizeL2 <- 200
sparsityParam <- 0.1
lambda <- 3e-3
beta <- 3

trainData <- loadImageFile('data/train-images-idx3-ubyte')
trainLabels <- loadLabelFile('data/train-labels-idx1-ubyte')

sae1Theta <- initializeParameters(hiddenSizeL1, inputSize)
sae1OptTheta <- optim(sae1Theta,
                    function(theta) sparseAutoencoderCost(theta, inputSize, hiddenSizeL1, lambda, sparsityParam, beta, trainData),
                    function(theta) sparseAutoencoderGrad(theta, inputSize, hiddenSizeL1, lambda, sparsityParam, beta, trainData),
                    method = "L-BFGS-B", control = list(trace = 3, maxit = 400))$par
sae1Features <- feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, inputSize, trainData)

sae2Theta <- initializeParameters(hiddenSizeL2, hiddenSizeL1)
sae2OptTheta <- optim(sae2Theta,
                      function(theta) sparseAutoencoderCost(theta, hiddenSizeL1, hiddenSizeL2, lambda, sparsityParam, beta, sae1Features),
                      function(theta) sparseAutoencoderGrad(theta, hiddenSizeL1, hiddenSizeL2, lambda, sparsityParam, beta, sae1Features),
                      method = "L-BFGS-B", control = list(trace = 3, maxit = 400))$par
sae2Features <- feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, hiddenSizeL1, sae1Features)

softmaxInputSize <- nrow(sae2Features)
numClasses <- length(table(trainLabels))
softmaxLambda <- 1e-4
softmaxTheta <- 0.005 * runif(hiddenSizeL2 * numClasses)
softmaxOptTheta <- optim(softmaxTheta,
                         function(theta) softmaxCost(theta, numClasses, softmaxInputSize, softmaxLambda, sae2Features, trainLabels),
                         function(theta) softmaxGrad(theta, numClasses, softmaxInputSize, softmaxLambda, sae2Features, trainLabels),
                         method = "L-BFGS-B", control = list(trace = 3, maxit = 400))$par

stackedAETheta <- c(softmaxOptTheta,
                    sae1OptTheta[1:(hiddenSizeL1 * inputSize + hiddenSizeL1)],
                    sae2OptTheta[1:(hiddenSizeL2 * hiddenSizeL1 + hiddenSizeL2)])

stackedAEOptTheta <- optim(stackedAETheta,
                           function(theta) stackedAECost(theta, inputSize, hiddenSizeL2, numClasses, lambda, trainData, trainLabels),
                           function(theta) stackedAEGrad(theta, inputSize, hiddenSizeL2, numClasses, lambda, trainData, trainLabels),
                           method = "L-BFGS-B", control = list(trace = 3, maxit = 400))$par


testData <- loadImageFile('data/t10k-images-idx3-ubyte')
testLabels <- loadLabelFile('data/t10k-labels-idx1-ubyte')

predict <- stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, numClasses, testData)
mean(predict == testLabels)
predict <- stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, numClasses, testData)
mean(predict == testLabels)

#****************************************** function **********************************************
stackedAECost <- function(theta, inputSize, hiddenSize, numClasses, lambda, data, labels) {
  softmaxTheta = matrix(theta[1:(hiddenSize*numClasses)], numClasses, hiddenSize)
  W1 <- matrix(theta[(hiddenSize*numClasses+1):(hiddenSize*numClasses+hiddenSize*inputSize)], hiddenSize, inputSize)
  b1 <- theta[(hiddenSize*numClasses+hiddenSize*inputSize+1):(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize)]
  W2 <- matrix(theta[(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+1):(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+hiddenSize*hiddenSize)], hiddenSize, hiddenSize)
  b2 <- theta[(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+hiddenSize*hiddenSize+1):(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+hiddenSize*hiddenSize+hiddenSize)]
  
  z2 <- W1 %*% data + b1
  a2 <- sigmoid(z2)
  z3 <- W2 %*% a2 + b2
  a3 <- sigmoid(z3)

  numCases <- ncol(data)
  groundTruth <- matrix(0, numClasses, numCases)
  for(i in 1:length(labels)) {
    groundTruth[labels[i], i] <- 1
  }

  M <- softmaxTheta %*% a3
  p <- apply(M, 2, function(x) {y <- x - max(x); return(exp(y) / sum(exp(y)))})
  cost <- -(1 / numCases) * sum(groundTruth * log(p)) + (lambda / 2) * sum(softmaxTheta^2)
  cost
}
stackedAEGrad <- function(theta, inputSize, hiddenSize, numClasses, lambda, data, labels) {
  softmaxTheta = matrix(theta[1:(hiddenSize*numClasses)], numClasses, hiddenSize)
  W1 <- matrix(theta[(hiddenSize*numClasses+1):(hiddenSize*numClasses+hiddenSize*inputSize)], hiddenSize, inputSize)
  b1 <- theta[(hiddenSize*numClasses+hiddenSize*inputSize+1):(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize)]
  W2 <- matrix(theta[(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+1):(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+hiddenSize*hiddenSize)], hiddenSize, hiddenSize)
  b2 <- theta[(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+hiddenSize*hiddenSize+1):(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+hiddenSize*hiddenSize+hiddenSize)]

  z2 <- W1 %*% data + b1
  a2 <- sigmoid(z2)
  z3 <- W2 %*% a2 + b2
  a3 <- sigmoid(z3)

  numCases <- ncol(data)
  groundTruth <- matrix(0, numClasses, numCases)
  for(i in 1:length(labels)) {
    groundTruth[labels[i], i] <- 1
  }

  M <- softmaxTheta %*% a3
  p <- apply(M, 2, function(x) {y <- x - max(x); return(exp(y) / sum(exp(y)))})
  softmaxThetagrad <- -(1 / numCases) * (groundTruth - p) %*% t(a3) + lambda * softmaxTheta

  delta3 <- -(t(softmaxTheta) %*% (groundTruth - p)) * a3 * (1 - a3)
  delta2 <- t(W2) %*% delta3 * a2 * (1 - a2)

  gradW2 <- (1 / numCases) * delta3 %*% t(a2)
  gradb2 <- (1 / numCases) * rowSums(delta3)
  gradW1 <- (1 / numCases) * delta2 %*% t(data)
  gradb1 <- (1 / numCases) * rowSums(delta2)
  c(as.vector(softmaxThetagrad),
    as.vector(gradW1),
    as.vector(gradb1),
    as.vector(gradW2),
    as.vector(gradb2))
}

stackedAEPredict <- function(theta, inputSize, hiddenSize, numClasses, data) {
  softmaxTheta = matrix(theta[1:(hiddenSize*numClasses)], numClasses, hiddenSize)
  W1 <- matrix(theta[(hiddenSize*numClasses+1):(hiddenSize*numClasses+hiddenSize*inputSize)], hiddenSize, inputSize)
  b1 <- theta[(hiddenSize*numClasses+hiddenSize*inputSize+1):(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize)]
  W2 <- matrix(theta[(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+1):(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+hiddenSize*hiddenSize)], hiddenSize, hiddenSize)
  b2 <- theta[(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+hiddenSize*hiddenSize+1):(hiddenSize*numClasses+hiddenSize*inputSize+hiddenSize+hiddenSize*hiddenSize+hiddenSize)]

  z2 <- W1 %*% data + b1
  a2 <- sigmoid(z2)
  z3 <- W2 %*% a2 + b2
  a3 <- sigmoid(z3)

  predict <- apply(softmaxTheta %*% a3, 2, function(x) which.max(x))
  predict
}
