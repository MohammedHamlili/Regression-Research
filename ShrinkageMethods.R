library(ISLR2)
library(dplyr)
# EDA of the dataset. We wish to predict a baseball player's salary based on 
# performance and other attributes, in order to better represent them at contract
# negotiations. 
attach(Hitters)
View(Hitters)
names(Hitters)
dim(Hitters) 
# 322 20
sum(is.na(Hitters$Salary))
# 59 points in the dataset have a null value in their salary category. 
Hitters = na.omit(Hitters)
dim(Hitters)
# 263 20

# Note: 20 predictors will induce a study of multicollinearity between all these
# predictors. This is where we will use our shrinkage methods in order to study
# the use of these methods and their effectiveness compared to traditional
# stepwise subset selection methods.

# Let's first fit the full model and analyze it.
full.model = lm(Salary ~ ., Hitters)
summary(full.model)

# Fitting Shrinkage methods.
library(glmnet)
set.seed(1006967631) 
# Splitting training and testing datasets.
# Will use 75% of data for training and 25% for testing. 
train = sample(1:nrow(Hitters), nrow(Hitters) * 0.75) 
x.train = Hitters[train, ]
x.train = select(x.train, -19) 
x.test = Hitters[-train, ]
x.test = select(x.test, -19)

y.train = Hitters$Salary[train]
y.test = Hitters$Salary[-train]

full.model.train = lm(Salary ~ ., data = Hitters[train,])
full.model.predict = predict(full.model.train, newdata = Hitters[-train,])
mean((y.test - full.model.predict) ^2)

# Let's fit the first shrinkage method: Ridge. This will shrink the coefficients
# of the previous model in order to account for multicollinearity and overfitting. 
# cv means we are using cross-validation. By default, cv uses 10-fold cross 
# validation to find the optimal value of the penalty term \lambda.
ridge.fit = cv.glmnet(data.matrix(x.train), y.train, type.measure = "mse", alpha = 0,
                      family = "gaussian")
ridge.predicted = predict(ridge.fit, s = ridge.fit$lambda.min, newx = data.matrix(x.test))
mean((y.test - ridge.predicted) ^ 2)


lasso.fit = cv.glmnet(data.matrix(x.train), y.train, type.measure = "mse", alpha = 1,
                      family = "gaussian")
lasso.predicted = predict(lasso.fit, s=lasso.fit$lambda.min, newx = data.matrix(x.test))
mean((y.test - lasso.predicted) ^ 2)

list.of.models = list()
for (i in 1:10) {
  model.name = paste0("alpha", i/10)
  list.of.models[[model.name]] = cv.glmnet(data.matrix(x.train), y.train, 
                                           type.measure = "mse", alpha = i/10,
                                           family = "gaussian")
  
}

result = data.frame()
for (i in 1:10) {
  model.name = paste0("alpha", i/10)
  model.predicted = predict(list.of.models[[model.name]], 
                            s = list.of.models[[model.name]]$lambda.min, 
                            newx = data.matrix(x.test))
  
  mse = mean((y.test - model.predicted) ^ 2)
  temp = data.frame(alpha = i/10, mse = mse, model.name = model.name)
  result = rbind(result, temp)
}


result
