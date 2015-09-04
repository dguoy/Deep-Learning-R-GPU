library(R6)

#************************************************ Function *******************************************************************************************
cnnConvolve <- function(patchDim, numFeatures, images, W, b, ZCAWhite, meanPatch) {
	numImages <- dim(images)[4]
	imageDim <- dim(images)[1]
	imageChannels <- dim(images)[3]
	convolvedDim <- imageDim - patchDim + 1
	
	convolvedFeatures <- array(0, c(convolvedDim, convolvedDim, numFeatures, numImages))
	
	WT <- W %*% ZCAWhite
	add <- b - WT %*% meanPatch
	
	for(imageNum in 1:numImages) {
		for(r in 1:convolvedDim) {
			for(c in 1:convolvedDim) {
				convolvedFeatures[r, c, , imageNum] <-
						sigmoid(WT %*% as.vector(images[r:(r+patchDim-1), c:(c+patchDim-1), ,imageNum]) + add)
			}
		}
	}
	convolvedFeatures
}
cnnPool <- function(poolDim, convolvedFeatures) {
	numImages <- dim(convolvedFeatures)[4]
	numFeatures <- dim(convolvedFeatures)[3]
	convolvedDim <- dim(convolvedFeatures)[1]
	poolLen <- convolvedDim / poolDim
	
	pooledFeatures <- array(0, c(poolLen, poolLen, numFeatures, numImages))
	for(i in 1 : numImages) {
		for(r in 1 : poolLen) {
			for(c in 1 : poolLen) {
				rb <- 1 + poolDim * (r-1)
				re <- poolDim * r
				cb <- 1 + poolDim * (c-1)
				ce <- poolDim * c
				pooledFeatures[r, c, , i] <- apply(matrix(convolvedFeatures[rb:re, cb:ce, ,i], poolDim * poolDim, numFeatures), MARGIN = 2, FUN = max)
			}
		}
	}
	pooledFeatures
}
#************************************************ With Object-oriented *******************************************************************************
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

Softmax <- R6Class("Softmax",
  private = list(
   numClasses=NA,
   inputSize=NA,
   lambda=NA,
   data=NA,
   labels=NA,
   numCases=NA,
   groundTruth=NA,
   M=NA,
   p=NA
  ),
  public = list(
   initialize = function(numClasses, inputSize, lambda, data, labels) {
     private$numClasses <- numClasses
     private$inputSize <- inputSize
     private$lambda <- lambda
     private$data <- data
     private$labels <- labels
   },
   cost = function(theta) {
     W = matrix(theta[1 : (private$numClasses * private$inputSize)], private$numClasses, private$inputSize)
     b = theta[(private$numClasses * private$inputSize + 1) : length(theta)]

     private$numCases <- ncol(private$data)

     private$groundTruth <- matrix(0, private$numClasses, private$numCases)
     for(i in 1:length(private$labels)) {
       private$groundTruth[private$labels[i], i] <- 1
     }

     private$M <- W %*% private$data + b
     private$p <- apply(private$M, 2, function(x) {y <- x - max(x); return(exp(y) / sum(exp(y)))})
     cost <- -(1 / private$numCases) * sum(private$groundTruth * log(private$p)) + (private$lambda / 2) * sum(W^2)
     return(cost)
   },
   grad = function(theta) {
     W = matrix(theta[1 : (private$numClasses * private$inputSize)], private$numClasses, private$inputSize)
     b = theta[(private$numClasses * private$inputSize + 1) : length(theta)]
     gradW <- -(1 / private$numCases) * (private$groundTruth - private$p) %*% t(private$data) + private$lambda * W
     gradb <- -(1 / private$numCases) * rowSums(private$groundTruth - private$p)
     return(c(gradW, gradb))
   },
   predict = function(theta, data, labels) {
     W = matrix(theta[1 : (private$numClasses * private$inputSize)], private$numClasses, private$inputSize)
     b = theta[(private$numClasses * private$inputSize + 1) : length(theta)]
     predict <- apply(W %*% data + b, 2, function(x) which.max(x))
     sum(predict == labels) / length(labels)
   }
  )
)
