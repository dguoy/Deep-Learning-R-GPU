PoolLayer <- R6Class("PoolLayer",
  private = list(
    input = NA,
    poolSize = NA,
    activation = NA
  ),
  public = list(
    initialize = function(input, poolSize, activation = 'max') {
      private$input <- input
      private$poolSize <- poolSize
      private$activation <- activation
    },
    output = function() {
      inputHeight <- dim(private$input)[1]
      inputWeight <- dim(private$input)[2]
      numFeatures <- dim(private$input)[3]
      numImages <- dim(private$input)[4]

      poolHeight <- private$poolSize[1]
      poolWeight <- private$poolSize[2]
      pooledHeight <- inputHeight / poolHeight
      pooledWeight <- inputWeight / poolWeight

      pooledFeatures <- array(0, c(pooledHeight, pooledWeight, numFeatures, numImages))
      for(i in 1:numImages) {
        for(r in 1:pooledHeight) {
          for(c in 1:pooledWeight) {
            rb <- 1 + poolHeight * (r-1)
            re <- poolHeight * r
            cb <- 1 + poolWeight * (c-1)
            ce <- poolWeight * c
            pooledFeatures[r, c, , i] <- apply(matrix(private$input[rb:re, cb:ce, ,i], poolHeight * poolWeight, numFeatures), MARGIN = 2, FUN = ifelse(private$activation == 'max', max, mean))
          }
        }
      }
      return (pooledFeatures)
    },
    getDelta = function(delta) {
      pooledHeight <- dim(delta)[1]
      pooledWeight <- dim(delta)[2]
      numFeatures <- dim(delta)[3]
      numImages <- dim(delta)[4]

      poolHeight <- private$poolSize[1]
      poolWeight <- private$poolSize[2]

      currentDelta <- array(0, c(pooledHeight * poolHeight, pooledWeight * poolWeight, numFeatures, numImages))
      if(private$activation == 'max') {
        for(i in 1:numImages) {
          for(r in 1:pooledHeight) {
            for(c in 1:pooledWeight) {
              rb <- 1 + poolHeight * (r-1)
              re <- poolHeight * r
              cb <- 1 + poolWeight * (c-1)
              ce <- poolWeight * c
              idx <- apply(matrix(private$input[rb:re, cb:ce, ,i], poolHeight * poolWeight, numFeatures), MARGIN = 2, FUN = which.max) + (0:(numFeatures - 1) * 4)
              currentDelta[rb:re, cb:ce, , i][idx] <- delta[r, c, , i]
            }
          }
        }
      } else {
        for(i in 1:numImages) {
          for(f in 1:numFeatures) {
            currentDelta[, , f, i] <- kronecker(delta[, , f, i], matrix(1 / (poolHeight * poolWeight), pooledHeight, pooledWeight))
          }
        }
      }
      return (currentDelta)
    }
  )
)

LeNetConvLayer <- R6Class("LeNetConvLayer",
  private = list(
    input = NA,
    filterShape = NA,
    convolvedFeatures = NA,
    W = NA,
    b = NA
  ),
  public = list(
    initialize = function(input, filterShape, W = NULL, b = NULL) {
      private$input <- input
      private$filterShape <- filterShape
      if(is.null(W)) {
        hiddenSize <- filterShape[3]
        inputSize <- filterShape[1] * filterShape[2] * dim(input)[3]
        r <- 4 * sqrt(6) / sqrt(hiddenSize+inputSize)
        private$W <- matrix(runif(hiddenSize * inputSize) * 2 * r - r, hiddenSize, inputSize)
      } else {
        private$W <- W
      }
      if(is.null(b)) {
        private$b <- rep(0, filterShape[3])
      } else {
        private$b <- b
      }
    },
    output = function() {
      if(is.na(private$convolvedFeatures)) {
        imageHeight <- dim(private$input)[1]
        imageWeight <- dim(private$input)[2]
        imageChannels <- dim(private$input)[3]
        numImages <- dim(private$input)[4]

        filterHeight <- private$filterShape[1]
        filterWeight <- private$filterShape[2]
        filterFeatures <- private$filterShape[3]

        convolvedHeight <- imageHeight - filterHeight + 1
        convolvedWeight <- imageWeight - filterWeight + 1

        private$convolvedFeatures <- array(0, c(convolvedHeight, convolvedWeight, filterFeatures, numImages))

        for(r in 1:convolvedHeight) {
          for(c in 1:convolvedWeight) {
            private$convolvedFeatures[r, c, , ] <-
              sigmoid(private$W %*% matrix(private$input[r:(r+filterHeight-1), c:(c+filterWeight-1), , ], ncol = numImages) + private$b)
          }
        }
      }
      
      return (private$convolvedFeatures)
    },
    grad = function(delta) {
      delta <- delta * private$convolvedFeatures * (1 - private$convolvedFeatures)

      convolvedHeight <- dim(delta)[1]
      convolvedWeight <- dim(delta)[2]
      numFeatures <- dim(private$input)[3]
      numImages <- dim(delta)[4]

      filterHeight <- private$filterShape[1]
      filterWeight <- private$filterShape[2]
      filterFeatures <- private$filterShape[3]

      gradW <- matrix(0, filterFeatures, filterHeight * filterWeight * numFeatures)
      gradb <- rep(0, filterFeatures)
      gradInput <- array(0, dim(private$input))
      for(r in 1:convolvedHeight) {
        for(c in 1:convolvedWeight) {
          gradW <- gradW +
            delta[r, c, , ] %*%
            matrix(private$input[r:(r+filterHeight-1), c:(c+filterWeight-1), , ], nrow = numImages, byrow = T)
          gradb <- gradb + rowSums(delta[r, c, , ])
          gradInput[r:(r+filterHeight-1), c:(c+filterWeight-1), , ] <- gradInput[r:(r+filterHeight-1), c:(c+filterWeight-1), , ] +
            array(t(W) %*% delta[r, c, , ], c(filterHeight, filterWeight, numFeatures, numImages))
        }
      }

      return (list('W' = gradW, 'b' = gradb, delta = gradInput))
    }
  )
)

HiddenLayer <- R6Class("HiddenLayer",
  private = list(
    activation = NA
  ),
  public = list(
    input = NA,
    inputSize = NA,
    outputSize = NA,
    W = NA,
    b = NA,
    ouput = NA,
    initialize = function(input, inputSize, outputSize, W, b, activation = sigmoid) {
      self$input <- input
      self$inputSize <- inputSize
      self$outputSize <- outputSize
      if(missing(W)) {
        r <- 4 * sqrt(6) / sqrt(inputSize + outputSize)
        self$W <- matrix(runif(inputSize * outputSize) * 2 * r - r, outputSize, inputSize)
      } else {
        self$W <- W
      }
      if(missing(b)) {
        self$b <- rep(0, outputSize)
      } else {
        self$b <- b
      }
      private$activation <- activation
    },
    output = function(input) {
      if(!missing(input)) self$input <- input
      linearOutput <- private$W %*% private$input + private$b
      if(is.null(private$activation)) {
        self$output <- linearOutput
      } else {
        self$output <- private$activation(linearOutput)
      }
      return (self$output)
    },
    grad = function(nextLayerDelta) {
      if(!is.null(private$activation)) {
        nextLayerDelta <- nextLayerDelta * private$output * (1 - private$output)
      }
      gradW <- nextLayerDelta %*% t(self$input)
      gradb <- rowSums(nextLayerDelta)
      delta <- t(W) %*% nextLayerDelta
      return (list('W' = gradW, 'b' = gradb, 'delta' = delta))
    }
  )
)

Softmax <- R6Class("Softmax",
  private = list(
    lambda = NA,
    groundTruth = NA,
    p = NA
  ),
  public = list(
    input = NA,
    y = NA,
    inputSize = NA,
    outputSize = NA,
    W = NA,
    b = NA,
    initialize = function(input, y, inputSize, outputSize, W, b, lambda = 0) {
      self$input <- input
      self$y <- y
      self$inputSize <- inputSize
      self$outputSize <- outputSize
      if (missing(W)) {
        self$W <- matrix(0.005 * runif(outputSize * inputSize), outputSize, inputSize)
      } else {
        self$W <- W
      }
      if (missing(b)) {
        self$b <- rep(0, outputSize)
      } else {
        self$b <- b
      }
      private$lambda <- lambda

      private$groundTruth <- matrix(0, outputSize, length(y))
      for(i in 1:length(y)) {
        private$groundTruth[y[i], i] <- 1
      }
    },
    cost = function() {
      private$p <- apply(self$W %*% self$input + self$b, 2, function(x) {y <- x - max(x); return(exp(y) / sum(exp(y)))})
      cost <- -(1 / length(self$y)) * sum(private$groundTruth * log(private$p)) + (private$lambda / 2) * sum(self$W^2)
      return (cost)
    },
    grad = function() {
      gradW <- -(1 / length(self$y)) * (private$groundTruth - private$p) %*% t(self$input) + private$lambda * self$W
      gradb <- -rowMeans(private$groundTruth - private$p)
      delta <- -(1 / length(self$y)) * t(self$W) %*% (private$groundTruth - private$p)
      return (list('W' = gradW, 'b' = gradb, 'delta' = delta))
    },
    train = function() {
      optimTheta <- optim(self$theta,
              function(theta) {self$theta <- theta; self$cost()},
              function(theta) {grad <- self$grad(); c(grad$W, grad$b)},
              method = "L-BFGS-B", control = list(trace = 3, maxit = maxIter))$par
    },
    output = function (input) {
      if(!missing(input)) self$input <- input
      return (apply(self$W %*% self$input + self$b, 2, function(x) which.max(x)))
    },
    errors = function(input, y) {
      return (mean(self$output(input) != y))
    }
  ),
  active = list(
    theta = function(theta) {
      if (missing(theta)) {
        return (c(self$W, self$b))
      } else {
        self$W <- matrix(theta[1 : (self$outputSize * self$inputSize)], self$outputSize, self$inputSize)
        self$b <- tail(theta, self$outputSize)
      }
    }
  )
)

SparseAutoencoder <- R6Class("SparseAutoencoder",
  private = list(
    input = NA,
	inputSize = NA,
    hiddenSize = NA,
    lambda = NA,
    sparsityParam = NA,
    beta = NA,
    layer1 = NA,
    layer2 = NA,
    rho = NA
  ),
  public = list(
    initialize = function(input, inputSize, hiddenSize, lambda, sparsityParam, beta) {
      private$input <- input
      private$inputSize <- inputSize
      private$lambda <- lambda
      private$sparsityParam <- sparsityParam
      private$beta <- beta
      layer1 <- HiddenLayer$new(input, inputSize, hiddenSize)
      layer2 <- HiddenLayer$new(layer1$output(), hiddenSize, inputSize)
    },
    cost = function(theta) {
      layer1$W = matrix(theta[1 : (private$hiddenSize*private$inputSize)], private$hiddenSize, private$inputSize)
      layer1$b = theta[(private$hiddenSize*private$inputSize+1) : (private$hiddenSize*private$inputSize+private$hiddenSize)]
      layer2$W = matrix(theta[(private$hiddenSize*private$inputSize+private$hiddenSize+1) : (2*private$hiddenSize*private$inputSize+private$hiddenSize)], private$inputSize, private$hiddenSize)
      layer2$b = theta[(2*private$hiddenSize*private$inputSize+private$hiddenSize+1) : length(theta)]

	  m <- ncol(private$input)

      a2 <- layer1$output()
      a3 <- layer2$output(a2)

      private$rho <- rowSums(a2) / m

      (1 /  (2 * m)) * sum((a3 - y)^2) +
          (private$lambda / 2) * (sum(private$lyaer2$W^2) + sum(private$lyaer3$W^2)) +
          private$beta * sum(private$sparsityParam * log(private$sparsityParam / private$rho) + (1 - private$sparsityParam) * log((1 - private$sparsityParam) / (1-private$rho)))
    },
    grad = function(theta) {
      m <- ncol(private$input)

      a2 <- layer1$output
      a3 <- layer2$output
	  
      sparsity_delta <- -private$sparsityParam / private$rho + (1-private$sparsityParam) / (1-private$rho)

      gradLayer2 <- layer2$grad(a3 - private$input)
      gradLayer1 <- layer1$grad(gradLayer2$delta + private$beta * sparsity_delta)

      W1grad <- (1 / m) * gradLayer1$W + private$lambda * layer1$W
      b1grad <- (1 / m) * gradLayer1$b
      W2grad <- (1 / m) * gradLayer2$W + private$lambda * layer2$W
      b2grad <- (1 / m) * gradLayer2$b
      
      c(as.vector(W1grad), as.vector(b1grad), as.vector(W2grad), as.vector(b2grad))
    }
  )
)
