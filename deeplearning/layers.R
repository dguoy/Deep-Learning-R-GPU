library(R6)

Softmax <- R6Class("Softmax",
  private = list(
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
    initialize = function(inputSize, outputSize, W, b) {
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
    },
    cost = function() {
      private$p <- apply(self$W %*% self$input + self$b, 2, function(x) {y <- x - max(x); return(exp(y) / sum(exp(y)))})
      cost <- -(1 / length(self$y)) * sum(private$groundTruth * log(private$p))
      return (cost)
    },
    grad = function() {
      gradW <- -(1 / length(self$y)) * (private$groundTruth - private$p) %*% t(self$input)
      gradb <- -rowMeans(private$groundTruth - private$p)
      delta <- -(1 / length(self$y)) * t(self$W) %*% (private$groundTruth - private$p)
      return (list('W' = gradW, 'b' = gradb, 'delta' = delta))
    },
    train = function(input, y, lambda, maxIter = 500) {
      self$input <- input
      self$y <- y
      private$groundTruth <- matrix(0, self$outputSize, length(y))
      for(i in 1:length(y)) {
        private$groundTruth[y[i], i] <- 1
      }
      optimTheta <- optim(self$theta,
              function(theta) {self$theta <- theta; self$cost() + (lambda / 2) * self$L2()},
              function(theta) {grad <- self$grad(); c(grad$W + lambda * self$W, grad$b)},
              method = "L-BFGS-B", control = list(trace = 3, maxit = maxIter))$par
      self$theta <- optimTheta
    },
    output = function(input) {
      return (apply(self$W %*% input + self$b, 2, which.max))
    },
    accuracy = function(input, y) {
      return (mean(self$output(input) == y))
    },
    L1 = function() {
      return (sum(abs(self$W)))
    },
    L2 = function() {
      return (sum(self$W^2))
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

HiddenLayer <- R6Class("HiddenLayer",
  private = list(
    activation = NA,
    finalOutput = NA
  ),
  public = list(
    input = NA,
    inputSize = NA,
    outputSize = NA,
    W = NA,
    b = NA,
    initialize = function(inputSize, outputSize, W, b, activation = sigmoid) {
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
      self$input <- input
      linearOutput <- self$W %*% self$input + self$b
      if(is.null(private$activation)) {
        private$finalOutput <- linearOutput
      } else {
        private$finalOutput <- private$activation(linearOutput)
      }
      return (private$finalOutput)
    },
    grad = function(nextLayerDelta) {
      if(!is.null(private$activation)) {
        nextLayerDelta <- nextLayerDelta * private$finalOutput * (1 - private$finalOutput)
      }
      gradW <- nextLayerDelta %*% t(self$input)
      gradb <- rowSums(nextLayerDelta)
      delta <- t(self$W) %*% nextLayerDelta
      return (list('W' = gradW, 'b' = gradb, 'delta' = delta))
    },
    L1 = function() {
      return (sum(abs(self$W)))
    },
    L2 = function() {
      return (sum(self$W^2))
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
    lambda = NA,
    sparsityParam = NA,
    beta = NA,
    rho = NA,
    finalOutput = NA
  ),
  public = list(
    input = NA,
    inputSize = NA,
    hiddenSize = NA,
    layer1 = NA,
    layer2 = NA,
    initialize = function(input, inputSize, hiddenSize, lambda, sparsityParam, beta, isLinearOutput = FALSE) {
      self$input <- input
      self$inputSize <- inputSize
      self$hiddenSize <- hiddenSize
      private$lambda <- lambda
      private$sparsityParam <- sparsityParam
      private$beta <- beta
      self$layer1 <- HiddenLayer$new(inputSize, hiddenSize)
      if(isLinearOutput) {
        self$layer2 <- HiddenLayer$new(hiddenSize, inputSize, activation = NULL)
      } else {
        self$layer2 <- HiddenLayer$new(hiddenSize, inputSize)
      }
    },
    cost = function() {
      m <- ncol(self$input)

      a2 <- self$layer1$output(self$input)
      private$finalOutput <- self$layer2$output(a2)

      private$rho <- rowSums(a2) / m

      (1 / (2 * m)) * sum((private$finalOutput - self$input)^2) +
          (private$lambda / 2) * (self$layer1$L2() + self$layer2$L2()) +
          private$beta * sum(private$sparsityParam * log(private$sparsityParam / private$rho) + (1 - private$sparsityParam) * log((1 - private$sparsityParam) / (1-private$rho)))
    },
    grad = function() {
      m <- ncol(self$input)

      sparsity_delta <- -private$sparsityParam / private$rho + (1-private$sparsityParam) / (1-private$rho)

      gradLayer2 <- self$layer2$grad((1 / m) * (private$finalOutput - self$input))
      gradLayer1 <- self$layer1$grad(gradLayer2$delta + (1 / m) * private$beta * sparsity_delta)

      return (list('W1' = gradLayer1$W, 'b1' = gradLayer1$b, 'W2' = gradLayer2$W, 'b2' = gradLayer2$b))
    },
    train = function(maxIter = 500) {
      optimTheta <- optim(self$theta,
              function(theta) {self$theta <- theta; self$cost()},
              function(theta) {grad <- self$grad(); c(grad$W1 + private$lambda * self$layer1$W, grad$b1, grad$W2 + private$lambda * self$layer2$W, grad$b2)},
              method = "L-BFGS-B", control = list(trace = 3, maxit = maxIter))$par
      self$theta <- optimTheta
    }
  ),
  active = list(
    theta = function(theta) {
      if (missing(theta)) {
        return (c(self$layer1$W, self$layer1$b, self$layer2$W, self$layer2$b))
      } else {
        self$layer1$theta <- theta[1 : (self$hiddenSize * self$inputSize + self$hiddenSize)]
        self$layer2$theta <- tail(theta, self$hiddenSize * self$inputSize + self$inputSize)
      }
    }
  )
)

Layers <- R6Class("Layers",
  private = list(
    layers = list()
  ),
  public = list(
    initialize = function(...) {
      for (layer in list(...)) {
        self$add(layer)
      }
    },
    output = function(input) {
      layerOutput <- input
      for (layer in private$layers) {
        layerOutput <- layer$output(layerOutput)
      }
      return (layerOutput)
    },
    grad = function(delta) {
      finalGrad = c()
      layerGrad <- list('delta' = delta)
      for (layer in rev(private$layers)) {
        layerGrad <- layer$grad(layerGrad$delta)
        finalGrad <- c(layerGrad$W, layerGrad$b, finalGrad)
      }
      return (finalGrad)
    },
    add = function(layer) {
      private$layers <- c(private$layers, list(layer))
      invisible(self)
    },
    depth = function() length(private$layers)
  )
)

PoolLayer <- R6Class("PoolLayer",
  private = list(
    activation = NA
  ),
  public = list(
    input = NA,
    poolSize = NA,
    initialize = function(poolSize, activation = 'max') {
      self$poolSize <- poolSize
      private$activation <- activation
    },
    output = function(input) {
      self$input <- inputb
      inputHeight <- dim(self$input)[1]
      inputWeight <- dim(self$input)[2]
      numFeatures <- dim(self$input)[3]
      numImages <- dim(self$input)[4]

      poolHeight <- self$poolSize[1]
      poolWeight <- self$poolSize[2]
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
            pooledFeatures[r, c, , i] <- apply(matrix(self$input[rb:re, cb:ce, ,i], poolHeight * poolWeight, numFeatures), MARGIN = 2, FUN = ifelse(private$activation == 'max', base::max, base::mean))
          }
        }
      }
      return (pooledFeatures)
    },
    grad = function(nextLayerDelta) {
      pooledHeight <- dim(nextLayerDelta)[1]
      pooledWeight <- dim(nextLayerDelta)[2]
      numFeatures <- dim(nextLayerDelta)[3]
      numImages <- dim(nextLayerDelta)[4]

      poolHeight <- self$poolSize[1]
      poolWeight <- self$poolSize[2]

      delta <- array(0, c(pooledHeight * poolHeight, pooledWeight * poolWeight, numFeatures, numImages))
      if(private$activation == 'max') {
        for(i in 1:numImages) {
          for(r in 1:pooledHeight) {
            for(c in 1:pooledWeight) {
              rb <- 1 + poolHeight * (r-1)
              re <- poolHeight * r
              cb <- 1 + poolWeight * (c-1)
              ce <- poolWeight * c
              idx <- apply(matrix(self$input[rb:re, cb:ce, ,i], poolHeight * poolWeight, numFeatures), MARGIN = 2, FUN = which.max) + (0:(numFeatures - 1) * 4)
              delta[rb:re, cb:ce, , i][idx] <- nextLayerDelta[r, c, , i]
            }
          }
        }
      } else {
        for(i in 1:numImages) {
          for(f in 1:numFeatures) {
            delta[, , f, i] <- kronecker(nextLayerDelta[, , f, i], matrix(1 / (poolHeight * poolWeight), pooledHeight, pooledWeight))
          }
        }
      }
      return (list('delta' = delta))
    }
  )
)

LeNetConvLayer <- R6Class("LeNetConvLayer",
  private = list(
  ),
  public = list(
    input = NA,
    filterShape = NA,
    convolvedFeatures = NA,
    W = NA,
    b = NA,
    initialize = function(filterShape, W, b) {
      self$filterShape <- filterShape
      if(missing(W)) {
        hiddenSize <- filterShape[3]
        inputSize <- filterShape[1] * filterShape[2] * dim(input)[3]
        r <- 4 * sqrt(6) / sqrt(hiddenSize+inputSize)
        self$W <- matrix(runif(hiddenSize * inputSize) * 2 * r - r, hiddenSize, inputSize)
      } else {
        self$W <- W
      }
      if(missing(b)) {
        self$b <- rep(0, filterShape[3])
      } else {
        self$b <- as.vector(b)
      }
    },
    output = function(input) {
      self$input <- input

      imageHeight <- dim(self$input)[1]
      imageWeight <- dim(self$input)[2]
      imageChannels <- dim(self$input)[3]
      numImages <- dim(self$input)[4]

      filterHeight <- self$filterShape[1]
      filterWeight <- self$filterShape[2]
      filterFeatures <- self$filterShape[3]

      convolvedHeight <- imageHeight - filterHeight + 1
      convolvedWeight <- imageWeight - filterWeight + 1

      self$convolvedFeatures <- array(0, c(convolvedHeight, convolvedWeight, filterFeatures, numImages))

      for(r in 1:convolvedHeight) {
        for(c in 1:convolvedWeight) {
          self$convolvedFeatures[r, c, , ] <-
            sigmoid(self$W %*% matrix(self$input[r:(r+filterHeight-1), c:(c+filterWeight-1), , ], ncol = numImages) + self$b)
        }
      }

      return (self$convolvedFeatures)
    },
    grad = function(nextLayerDelta) {
      nextLayerDelta <- nextLayerDelta * self$convolvedFeatures * (1 - self$convolvedFeatures)

      convolvedHeight <- dim(nextLayerDelta)[1]
      convolvedWeight <- dim(nextLayerDelta)[2]
      numFeatures <- dim(self$input)[3]
      numImages <- dim(nextLayerDelta)[4]

      filterHeight <- self$filterShape[1]
      filterWeight <- self$filterShape[2]
      filterFeatures <- self$filterShape[3]

      gradW <- matrix(0, filterFeatures, filterHeight * filterWeight * numFeatures)
      gradb <- rep(0, filterFeatures)
      delta <- array(0, dim(self$input))
      for(r in 1:convolvedHeight) {
        for(c in 1:convolvedWeight) {
          gradW <- gradW +
            nextLayerDelta[r, c, , ] %*%
            matrix(self$input[r:(r+filterHeight-1), c:(c+filterWeight-1), , ], nrow = numImages, byrow = T)
          gradb <- gradb + rowSums(nextLayerDelta[r, c, , ])
          delta[r:(r+filterHeight-1), c:(c+filterWeight-1), , ] <- delta[r:(r+filterHeight-1), c:(c+filterWeight-1), , ] +
            array(t(W) %*% nextLayerDelta[r, c, , ], c(filterHeight, filterWeight, numFeatures, numImages))
        }
      }

      return (list('W' = gradW, 'b' = gradb, 'delta' = delta))
    }
  )
)
