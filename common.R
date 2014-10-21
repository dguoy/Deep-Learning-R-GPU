library(R.matlab)
library(reshape2)
library(ggplot2)
library(R6)

initializeParameters <- function(hiddenSize, visibleSize) {
	r <- sqrt(6) / sqrt(hiddenSize+visibleSize+1)
	W1 <- runif(hiddenSize * visibleSize) * 2 * r - r
	b1 <- rep(0, hiddenSize)
	W2 <- runif(visibleSize * hiddenSize) * 2 * r - r
	b2 <- rep(0, visibleSize)
	c(W1, b1, W2, b2)
}
sigmoid <- function(z) 1 / (1 + exp(-z))
sigmoidRnd <- function(z) {
  ifelse(sigmoid(z) > matrix(runif(length(z)), nrow = nrow(z), ncol = ncol(z)), 1, 0)
}
feedForwardAutoencoder <- function(theta, hiddenSize, visibleSize, data) {
	W1 <- matrix(theta[1:(hiddenSize*visibleSize)], hiddenSize, visibleSize)
	b1 <- theta[(hiddenSize*visibleSize+1) : (hiddenSize*visibleSize+hiddenSize)]
	z2 <- W1 %*% data + b1
	a2 <- sigmoid(z2)
	a2
}
sampleImages <- function(images, patchsize=8, numpatches=100000) {
	imageRowDim <- dim(images)[1]
	imageColDim <- dim(images)[2]
	numImages <- dim(images)[3]
	patches <- matrix(0, nrow=patchsize^2, ncol=numpatches)
	for(i in 1:numpatches) {
		x <- sample(1:(imageRowDim-patchsize+1), size=1)
		y <- sample(1:(imageColDim-patchsize+1), size=1)
		z <- sample(1:numImages, size=1)
		patches[, i] <- as.vector(images[x:(x+patchsize-1), y:(y+patchsize-1), z])
	}
	patches <- patches - mean(patches)
	pstd <- 3 * sd(patches)
	patches[patches > pstd] <- pstd
	patches[patches < -pstd] <- -pstd
	patches <- (patches / pstd + 1) * 0.4 + 0.1
	patches
}
sampleColorImages <- function(images, patchsize=8, numpatches=100000) {
	imageRowDim <- dim(images)[1]
	imageColDim <- dim(images)[2]
	imageChannels <- dim(images)[3]
	numImages <- dim(images)[4]
	patches <- matrix(0, nrow=patchsize^2 * imageChannels, ncol=numpatches)
	for(i in 1:numpatches) {
		x <- sample(1:(imageRowDim-patchsize+1), size=1)
		y <- sample(1:(imageColDim-patchsize+1), size=1)
		z <- sample(1:numImages, size=1)
		patches[, i] <- as.vector(images[x:(x+patchsize-1), y:(y+patchsize-1), , z])
	}
	patches
}
rotate <- function(x) t(apply(x, 2, rev))
displayColorNetwork <- function(A) {
	if(min(A) >= 0) {
		A <- A - mean(A)
	}

	cols = round(sqrt(ncol(A)))
	channelSize <- nrow(A) / 3
	dim <- sqrt(channelSize)
	dimp <- dim + 1
	rows <- ceiling(ncol(A) / cols)
	B = A[1:channelSize,]
	C = A[(channelSize+1):(channelSize*2),]
	D = A[(2*channelSize+1):(channelSize*3),]
	B <- B / max(abs(B))
	C <- C / max(abs(C))
	D <- D / max(abs(D))
	I <- array(1, c(dim*rows+rows-1, dim*cols+cols-1, 3))
	
	for(i in 0:(rows-1)) {
		for(j in 0:(cols-1)) {
			I[(i*dimp+1):(i*dimp+dim), (j*dimp+1):(j*dimp+dim), 1] <- 
					rotate(matrix(B[, i*cols+j+1],dim, dim))
			I[(i*dimp+1):(i*dimp+dim), (j*dimp+1):(j*dimp+dim), 2] <- 
					rotate(matrix(C[, i*cols+j+1],dim, dim))
			I[(i*dimp+1):(i*dimp+dim), (j*dimp+1):(j*dimp+dim), 3] <- 
					rotate(matrix(D[, i*cols+j+1],dim, dim))
		}
	}
	I <- (I + 1) / 2

	R <- melt(I[,, 1])
	G <- melt(I[,, 2])
	B <- melt(I[,, 3])
	displayImage <- merge(R, G, by.x=c('Var1', 'Var2'), by.y = c("Var1", "Var2"), all = TRUE)
	displayImage <- merge(displayImage, B, by.x=c('Var1', 'Var2'), by.y = c("Var1", "Var2"), all = TRUE)
	colnames(displayImage) <- c('x', 'y', 'r', 'g', 'b')
	
	ggplot(data=as.data.frame(displayImage), aes(x=x, y=y, fill=rgb(r,g,b))) +
			geom_tile() +
			scale_fill_identity()
}
up2Down <- function(A) A[,ncol(A):1]
displayNetwork <- function(A) {
  if(min(A) <= 0) {
    A <- A - min(A)
  }
  cols = round(sqrt(nrow(A)))
  rows <- ceiling(nrow(A) / cols)
  dim <- sqrt(ncol(A))
  dimp <- dim + 1
  I <- matrix(0, dim*rows+rows-1, dim*cols+cols-1)

  for(i in 0:(rows-1)) {
    for(j in 0:(cols-1)) {
      I[(i*dimp+1):(i*dimp+dim), (j*dimp+1):(j*dimp+dim)] <- 
        up2Down(matrix(A[i*cols+j+1, ] / sqrt(sum(A[i*cols+j+1, ]^2)), dim, dim))
    }
  }

  ggplot(melt(I), aes(Var1, Var2, fill=value)) +
    geom_tile() +
    scale_fill_gradient(low="#FFFFFF", high="#000000")
}
checkNumericalGradient <- function(theta, f, g) {
	actualGrad <- g(theta)
	epsilon <- 10^-4
	gradient <- rep(0, length(theta))
	for(i in 1:length(theta)) {
		e <- rep(0, length(theta))
		e[i] <- epsilon
		gradient[i] <- (f(theta + e) - f(theta - e)) / (2 * epsilon)
		if(abs(gradient[i] - actualGrad[i]) > 1e-9) {
		  print(sprintf("%s = %s", i, abs(gradient[i] - actualGrad[i])))
		}
	}
	sum(abs(gradient - actualGrad))
}
loadImageFile <- function(filename) {
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  n = readBin(f,'integer',n=1,size=4,endian='big')
  nrow = readBin(f,'integer',n=1,size=4,endian='big')
  ncol = readBin(f,'integer',n=1,size=4,endian='big')
  x = readBin(f,'integer',n=n*nrow*ncol,size=1,signed=F)
  ret = matrix(x, nrow=nrow*ncol) / 255
  close(f)
  ret
}
loadLabelFile <- function(filename) {
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  n = readBin(f,'integer',n=1,size=4,endian='big')
  y = readBin(f,'integer',n=n,size=1,signed=F)
  close(f)
  y[y == 0] <- 10
  y
}

