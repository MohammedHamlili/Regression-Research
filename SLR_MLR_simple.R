# First part of the study:
#   Simple Linear Regression and Multi Linear Regression Analysis 
# This analysis is done with a dataset that allows for a simple analysis in order
# to motivate the further study of high dimensional regressional methods.
 
# We can use the "mtcars" dataset which is a built-in R dataset.
data(mtcars)
# A useful study here would be to analyze how various features of the car impact
# the miles that can be covered per gallon of gas. This can be a useful study to
# improve the gas usage which has an obviously important environmental effect.

# To fit a linear model we can use the lm() function.
model = lm(mpg ~ ., data = mtcars)
# Notice that the "~ ." is the R command to fit all the variables of the 
# dataset against mpg.

mod = summary(model)
# summary of the model.

cor(mtcars)
library(car)
vif(model)
# Gets us VIF values. QUESTION : how to interpret these (==> how to choose
# correlated variables) ?

## Cutoff: Subjective, but usually > 10. Actually need to eliminate variables
## with highest VIF values. 

interaction.plot(mtcars$wt, mtcars$disp, mtcars$mpg) # maybe useless (only 1 line)
# QUESTION: How to interpret these plots ? 
pairs(mtcars)

# Run backward stepwise regression: Evaluate fitting the regression with one less variable
# with each variable being removed and then test the model and select the best model with one
# less variable. Keep going until it becomes worse to remove a variable.
step(model, direction = "backward")


model1 = lm(mpg ~ wt + qsec + am, data = mtcars)
mod1 = summary(model1)
# If we look at the correlation matrix, we can see wt is highly correlated with disp
# and am is highly correlated with gear. We can try fitting another model with these 
# 2 predictors respectively replaced.
model2 = lm(mpg ~ disp + qsec + am, data = mtcars)
mod2 = summary(model2)
# We can test another model with one less predictor to test how that one compares
# with the models obtained previously using backward stepwise regression.
model3 = lm(mpg ~ wt + hp, data = mtcars)
mod3 = summary(model3)

# Different models to introduce the concept of multi linear regression and 
# the problem of choosing the best predictors.
# model1 = lm(mpg ~ disp + wt, data = mtcars)
# mod1 = summary(model1)
# model2 = lm(mpg~wt+hp, data=mtcars)
# mod2 = summary(model2)
# QUESTION: Different p_value for wt. What does it mean?

plot(model)
plot(model1)
plot(model2)
plot(model3)
## QUESTION: how can we use these plots to discuss the point we are trying
## to make about the multicollinearity of multiple predictors?

set.seed(192029382)
mtcars$rand = runif(nrow(mtcars))
## Splitting the dataset into train and test sets to train the model then test
# it. We'll do this for all 4 models we have created and compare them.

trainIndex = which(mtcars$rand <= 0.8)
train = mtcars[trainIndex, ]
test = mtcars[-trainIndex, ]

# Refitting the models on just the train datasets.
model = lm(mpg ~ ., data = train)
model1 = lm(mpg ~ wt + qsec + am, data = train)
model2 = lm(mpg ~ disp + qsec + am, data = train)
model3 = lm(mpg ~ wt + hp, data = train)

predictions = predict(model, test)
predictions1 = predict(model1, test)
predictions2 = predict(model2, test)
predictions3 = predict(model3, test)

mse = mean((predictions - test$mpg) ^ 2)
mse1 = mean((predictions1 - test$mpg) ^ 2)
mse2 = mean((predictions2 - test$mpg) ^ 2)
mse3 = mean((predictions3 - test$mpg) ^ 2)


# So, it is clear that the best model is the last one where we fitted with 
# wt and hp. However, it is complicated to choose between among all predictors
# which ones will be the most useful and will lead to the better fit, other than
# having to try each combination which can become computationally very complex
# once we have a larger number of predictors. Therefore, the goal of this paper
# is to introduce other methods of high dimensional multi linear regression
# that allow to get around this difficulty. Mainly, we will looking at 
# shrinking and regularization methods such as Lasso and Ridge.


