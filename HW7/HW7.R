#############################################################
## Stat 202A - Homework 7
## Author: Chaojie Feng
## Date : Dec.4, 2018
## Description: This script implements the lasso
#############################################################

#############################################################
## INSTRUCTIONS: Please fill in the missing lines of code
## only where specified. Do not change function names,
## function inputs or outputs. You can add examples at the
## end of the script (in the "Optional examples" section) to
## double-check your work, but MAKE SURE TO COMMENT OUT ALL
## OF YOUR EXAMPLES BEFORE SUBMITTING.
##
## Very important: Do not use the function "setwd" anywhere
## in your code. If you do, I will be unable to grade your
## work since R will attempt to change my working directory
## to one that does not exist.
#############################################################

## Source your Rcpp file (put in the name of your Rcpp file)
library(Rcpp)
sourceCpp("./HW7.cpp")

##################################
## Function 1: Ridge regression ##
##################################

myRidge <- function(X, Y, lambda, use_QR = FALSE, use_C = TRUE){
  
  # Perform ridge regression of Y on X.
  # 
  # X: an n x p matrix of explanatory variables.
  # Y: an n vector of dependent variables. Y can also be a 
  # matrix, as long as the function works.
  # lambda: regularization parameter (lambda >= 0)
  # Returns beta, the ridge regression solution.
  
  ##################################
  ## FILL IN THIS SECTION OF CODE ##
  ##################################
  n = dim(X)[1]
  p = dim(X)[2]
  
  z = cbind(rep(1,n), X, Y)
  
  A = t(z) %*% z
  
  D = diag(rep(lambda, p+2))
  D[1,1] = 0
  D[p+2,p+2] = 0
  A = A + D
  S = mySweepC(A, p+1)
  beta_ridge =S[1:(p+1), p+2]
  
  
  ## Function should output the vector beta_ridge, the 
  ## solution to the ridge regression problem. beta_ridge
  ## should have p + 1 elements.
  return(beta_ridge)
  
}

####################################################
## Function 2: Piecewise linear spline regression ##
####################################################

mySpline <- function(x, Y, lambda = 1, p = 100, use_QR = FALSE, use_C = TRUE){
  
  # Perform spline regression of Y on X.
  # 
  # x: An n x 1 vector or matrix of explanatory variables.
  # Y: An n x 1 vector of dependent variables. Y can also be a 
  # matrix, as long as the function works.
  # lambda: regularization parameter (lambda >= 0)
  # p: Number of cuts to make to the x-axis.
  
  ##################################
  ## FILL IN THIS SECTION OF CODE ##
  ##################################
  n = length(x)
  
  X = matrix(x, nrow=n)
  for (k in (1:(p-1))/p){
    X = cbind(X, (x>k)*(x-k))
  }
  
  beta_spline = myRidge(X, Y, lambda)
  Yhat = cbind(rep(1,n), X) %*% beta_spline
  
  ## Function should a list containing two elements:
  ## The first element of the list is the spline regression
  ## beta vector, which should be p + 1 dimensional (here, 
  ## p is the number of cuts we made to the x-axis).
  ## The second element is y.hat, the predicted Y values
  ## using the spline regression beta vector. This 
  ## can be a numeric vector or matrix.
  output <- list(beta_spline = beta_spline, predicted_y = Yhat)
  return(output)
  
}

#####################################
##   Function 3: Visualizations    ##
#####################################

myPlotting <- function(){

  # Write code to plot the result. 

  #######################
  ## FILL IN CODE HERE ##
  #######################
  
  ################
  # Ridge and Spline
  ################
  
  n = 40
  p = 600
  sigma = .2
  lambda = 1.
  x = runif(n)
  x = sort(x)
  Y = 4*(x - 0.5)^2 + (rnorm(n))*sigma
  
  output1 <- mySpline(x, Y, lambda = 0.01)
  output2 <- mySpline(x, Y, lambda = 0.1)
  output3 <- mySpline(x, Y, lambda = 1)
  output4 <- mySpline(x, Y, lambda = 10)
  output5 <- mySpline(x, Y, lambda = 100)
  
  yhat = list(output1$predicted_y, output2$predicted_y, output3$predicted_y, output4$predicted_y, output5$predicted_y)
  plot(x, Y, col = "darkgreen", pch = 24)
  lines(x, yhat[[1]], type = "l", col = "green")
  lines(x, yhat[[2]], type = "l", col = "yellow")
  lines(x, yhat[[3]], type = "l", col = "blue")
  lines(x, yhat[[4]], type = "l", col = "red")
  lines(x, yhat[[5]], type = "l", col = "black")
  
  
  legend("topleft", legend=c("lambda = 0.01", "lambda = 0.1", "lambda = 1", "lambda = 10", "lambda = 100"),
         col=c("green", "yellow", "blue", "red", "black"), lty=c(1,1,1,1,1), cex=0.8)
  title(main="Spline Test", 
        xlab="x", ylab="y")
  
  lambda_array = c(0.01, 0.1, 1, 10, 100)
  i = 1
  err = c()
  for(i in 1:5){
    err[i] = sum((Y - yhat[[i]])^2)/n
  }
  
  plot(lambda_array, err, col="red")
  title("Error plot for different lambda")
  

  
  ###################
  # Lasso and Boosting
  ###################

  n = 50
  p = 200
  lambda_all = (100:1) * 10
  lambda_all = matrix(lambda_all)
  L = length(lambda_all)
  
  X = matrix(rnorm(n*p), nrow = n)
  beta_true = matrix(rep(0,p), nrow = p)
  beta_true[1:5] = 1:5
  Y = 1 + X %*% beta_true + rnorm(n)
  
  # lasso solution
  beta_all_lasso = myLassoC(X, Y, lambda_all)
  
  # e-boosting solution
  beta_all_eboost = myBoostingC(X, Y, lambda_all)
  
  
  
  # lasso path
  plt_lasso <- matplot(t(matrix(rep(1,p+1),nrow=1)%*%abs(beta_all_lasso)), t(beta_all_lasso), type = 'l', xlab="l1", ylab="beta")
  title(main="Lasso solution path")
  
  # epsilon path
  plt_epsilon <- matplot(t(matrix(rep(1,p+1),nrow=1)%*%abs(beta_all_eboost)), t(beta_all_eboost), type = 'l', xlab="l1", ylab="beta")
  title(main="Epsilon-boosting solution path")
  
}



