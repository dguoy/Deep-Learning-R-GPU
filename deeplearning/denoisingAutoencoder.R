library(R6)

#************************************************ Function *******************************************************************************************
denoisingAutoencoderCost <- function(theta, visibleSize, hiddenSize, data) {
  W = matrix(theta[1 : (hiddenSize*visibleSize)], hiddenSize, visibleSize)
  bhid = theta[(hiddenSize*visibleSize+1) : (hiddenSize*visibleSize+hiddenSize)]
  bvis = theta[(hiddenSize*visibleSize+hiddenSize+1) : length(theta)]

  z2 <- W %*% data + bhid
  a2 <- sigmoid(z2)
  z3 <- t(W) %*% a2 + bvis
  a3 <- sigmoid(z3)

  L <- -colSums(data * log(a3) + (1 - data) * log(1 - a3))
  return (mean(L))
}
denoisingAutoencoderGrad <- function(theta, visibleSize, hiddenSize, data) {
  W = matrix(theta[1 : (hiddenSize*visibleSize)], hiddenSize, visibleSize)
  bhid = theta[(hiddenSize*visibleSize+1) : (hiddenSize*visibleSize+hiddenSize)]
  bvis = theta[(hiddenSize*visibleSize+hiddenSize+1) : length(theta)]

  z2 <- W %*% data + bhid
  a2 <- sigmoid(z2)
  z3 <- t(W) %*% a2 + bvis
  a3 <- sigmoid(z3)

  m <- ncol(data)

  LGrad <- data / a3 - (1 - data) / (1 - a3)
  a3Grad <- a3 * (1 - a3)
  a2Grad <- a2 * (1 - a2)
  a3GradLGrad <- a3Grad * LGrad

  WGrad <- -(a2Grad * (W %*% a3GradLGrad)) %*% t(data) - a2 %*% t(a3GradLGrad)
  WGrad <- WGrad / m

  bhidGrad <- -rowMeans((W %*% a3GradLGrad) * a2Grad)
  bvisGrad <- -rowMeans(a3GradLGrad)

  return (c(WGrad, bhidGrad, bvisGrad))
}
#************************************************ With Object-oriented *******************************************************************************
