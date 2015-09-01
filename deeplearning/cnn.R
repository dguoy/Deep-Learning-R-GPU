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
    initialize = function(input, poolSize, activation = max) {
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
            pooledFeatures[r, c, , i] <- apply(matrix(private$input[rb:re, cb:ce, ,i], poolHeight * poolWeight, numFeatures), MARGIN = 2, FUN = private$activation)
          }
        }
      }
      return (pooledFeatures)
    }
  )
)

LeNetConvLayer <- R6Class("LeNetConvLayer",
  private = list(
  　input = NA,
  　filterShape = NA,
  　W = NA,
  　b = NA
  ),
  public = list(
   initialize = function(input, filterShape, W, b) {
     private$input <- input
     private$filterShape <- filterShape
     private$W <- W
     private$b <- b
   },
   output = function() {
     imageHeight <- dim(private$input)[1]
     imageWeight <- dim(private$input)[2]
     imageChannels <- dim(private$input)[3]
     numImages <- dim(private$input)[4]

     filterHeight <- private$filterShape[1]
     filterWeight <- private$filterShape[2]
     filterFeatures <- private$filterShape[3]

     convolvedHeight <- imageHeight - filterHeight + 1
     convolvedWeight <- imageWeight - filterWeight + 1

     convolvedFeatures <- array(0, c(convolvedHeight, convolvedWeight, filterFeatures, numImages))

     for(r in 1:convolvedHeight) {
       for(c in 1:convolvedWeight) {
         convolvedFeatures[r, c, , ] <-
           sigmoid(private$W %*% matrix(private$input[r:(r+filterHeight-1), c:(c+filterWeight-1), , ], ncol = numImages) + private$b)
       }
     }

     return (convolvedFeatures)
   }
  )
)
