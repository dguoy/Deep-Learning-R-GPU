library(R6)

#*****************************************************************************************************************************************************
sparseAutoencoderLinearCost <- function(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data) {
	W1 = matrix(theta[1 : (hiddenSize*visibleSize)], hiddenSize, visibleSize)
	b1 = theta[(hiddenSize*visibleSize+1) : (hiddenSize*visibleSize+hiddenSize)]
	W2 = matrix(theta[(hiddenSize*visibleSize+hiddenSize+1) : (2*hiddenSize*visibleSize+hiddenSize)], visibleSize, hiddenSize)
	b2 = theta[(2*hiddenSize*visibleSize+hiddenSize+1) : length(theta)]

	m <- ncol(data)
	y <- data

	z2 <- W1 %*% data + b1
	a2 <- sigmoid(z2)
	z3 <- W2 %*% a2 + b2
	a3 <- z3

	rho <- rowMeans(a2)

	(1 /  (2 * m)) * sum((a3 - y)^2) +
		(lambda / 2) * (sum(W1^2) + sum(W2^2)) +
			beta * sum(sparsityParam * log(sparsityParam / rho) + (1 - sparsityParam) * log((1 - sparsityParam) / (1-rho)))
}
sparseAutoencoderLinearGrad <- function(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data) {
	W1 = matrix(theta[1 : (hiddenSize*visibleSize)], hiddenSize, visibleSize)
	b1 = theta[(hiddenSize*visibleSize+1) : (hiddenSize*visibleSize+hiddenSize)]
	W2 = matrix(theta[(hiddenSize*visibleSize+hiddenSize+1) : (2*hiddenSize*visibleSize+hiddenSize)], visibleSize, hiddenSize)
	b2 = theta[(2*hiddenSize*visibleSize+hiddenSize+1) : length(theta)]

	m <- ncol(data)
	y <- data

	z2 <- W1 %*% data + b1
	a2 <- sigmoid(z2)
	z3 <- W2 %*% a2 + b2
	a3 <- z3

	rho <- rowMeans(a2)

	sparsity_delta <- -sparsityParam / rho + (1-sparsityParam) / (1-rho)

	delta3 <- -(y - a3)
	delta2 <- (t(W2) %*% delta3 + beta * sparsity_delta) * a2 * (1 - a2)

	deltaW1 <- delta2 %*% t(data)
	deltab1 <- rowSums(delta2)
	deltaW2 <- delta3 %*% t(a2)
	deltab2 <- rowSums(delta3)

	W1grad <- (1 / m) * deltaW1 + lambda * W1
	b1grad <- (1 / m) * deltab1
	W2grad <- (1 / m) * deltaW2 + lambda * W2
	b2grad <- (1 / m) * deltab2

	c(as.vector(W1grad), as.vector(b1grad), as.vector(W2grad), as.vector(b2grad))
}
#************************************************ With Object-oriented *******************************************************************************
SparseAutoencoderLinear <- R6Class("SparseAutoencoderLinear",
	private = list(
		visibleSize=NA,
		hiddenSize=NA,
		lambda=NA,
		sparsityParam=NA,
		beta=NA,
		data=NA,
		a2=NA,
		a3=NA,
		rho=NA
	),
	public = list(
		initialize = function(visibleSize, hiddenSize, lambda, sparsityParam, beta, data) {
			private$visibleSize <- visibleSize
			private$hiddenSize <- hiddenSize
			private$lambda <- lambda
			private$sparsityParam <- sparsityParam
			private$beta <- beta
			private$data <- data
		},
		cost = function(theta) {
			W1 = matrix(theta[1 : (private$hiddenSize*private$visibleSize)], private$hiddenSize, private$visibleSize)
			b1 = theta[(private$hiddenSize*private$visibleSize+1) : (private$hiddenSize*private$visibleSize+private$hiddenSize)]
			W2 = matrix(theta[(private$hiddenSize*private$visibleSize+private$hiddenSize+1) : (2*private$hiddenSize*private$visibleSize+private$hiddenSize)], private$visibleSize, private$hiddenSize)
			b2 = theta[(2*private$hiddenSize*private$visibleSize+private$hiddenSize+1) : length(theta)]

			m <- ncol(private$data)
			y <- private$data

			z2 <- W1 %**% private$data + b1
			private$a2 <- sigmoid(z2)
			z3 <- W2 %**% private$a2 + b2
			private$a3 <- z3

			private$rho <- rowSums(private$a2) / m

			(1 /  (2 * m)) * sum((private$a3 - y)^2) +
				(private$lambda / 2) * (sum(W1^2) + sum(W2^2)) +
			  private$beta * sum(private$sparsityParam * log(private$sparsityParam / private$rho) + (1 - private$sparsityParam) * log((1 - private$sparsityParam) / (1-private$rho)))
		},
		grad = function(theta) {
			W1 = matrix(theta[1 : (private$hiddenSize*private$visibleSize)], private$hiddenSize, private$visibleSize)
			b1 = theta[(private$hiddenSize*private$visibleSize+1) : (private$hiddenSize*private$visibleSize+private$hiddenSize)]
			W2 = matrix(theta[(private$hiddenSize*private$visibleSize+private$hiddenSize+1) : (2*private$hiddenSize*private$visibleSize+private$hiddenSize)], private$visibleSize, private$hiddenSize)
			b2 = theta[(2*private$hiddenSize*private$visibleSize+private$hiddenSize+1) : length(theta)]

			m <- ncol(private$data)
			y <- private$data

			sparsity_delta <- -private$sparsityParam / private$rho + (1-private$sparsityParam) / (1-private$rho)

			delta3 <- -(y - private$a3)
			delta2 <- (t(W2) %**% delta3 + private$beta * sparsity_delta) * private$a2 * (1 - private$a2)

			deltaW1 <- delta2 %**% t(private$data)
			deltab1 <- rowSums(delta2)
			deltaW2 <- delta3 %**% t(private$a2)
			deltab2 <- rowSums(delta3)

			W1grad <- (1 / m) * deltaW1 + private$lambda * W1
			b1grad <- (1 / m) * deltab1
			W2grad <- (1 / m) * deltaW2 + private$lambda * W2
			b2grad <- (1 / m) * deltab2

			c(as.vector(W1grad), as.vector(b1grad), as.vector(W2grad), as.vector(b2grad))
		}
	)
)
