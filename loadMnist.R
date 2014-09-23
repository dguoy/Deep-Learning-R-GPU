loadImageFile <- function(filename) {
	ret = list()
	f = file(filename,'rb')
	readBin(f,'integer',n=1,size=4,endian='big')
	ret$n = readBin(f,'integer',n=1,size=4,endian='big')
	nrow = readBin(f,'integer',n=1,size=4,endian='big')
	ncol = readBin(f,'integer',n=1,size=4,endian='big')
	x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
	ret$x = matrix(x, nrow=nrow*ncol) / 255
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
train <- loadImageFile('data/train-images-idx3-ubyte')
test <- loadImageFile('data/t10k-images-idx3-ubyte')
train$y <- loadLabelFile('data/train-labels-idx1-ubyte')
test$y <- loadLabelFile('data/t10k-labels-idx1-ubyte')

