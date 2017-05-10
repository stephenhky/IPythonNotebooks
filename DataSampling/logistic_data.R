
decide<- function(W, X) as.matrix(X) %*% as.matrix(W)

W<- c(1, -2)
X<- data.frame(X1=runif(50, min=-5, max=5), X2=runif(50, min=-5, max=5))

df<- data.frame(X, Y=ifelse(decide(W, X)+rnorm(50, sd = 5)>0, 1, 0))


write.csv(df, 'logistic_test.csv', row.names=FALSE)

lmodel<- glm(Y~X1+X2, data=df, family = 'binomial')

