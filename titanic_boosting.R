
library(dplyr)
library(tidyr)

library(rpart)
library(Ckmeans.1d.dp)



train <- read.csv("train.csv")

test <- read.csv("test.csv")

train <- train%>%
  select(2, 3, 5, 6, 7, 8, 10, 12)

train <- na.omit(train)

str(train)

train <- train%>%
  mutate(Survived = as.numeric(Survived),
         Pclass = as.numeric(Pclass),
         Sex = as.factor(Sex),
         Age = as.numeric(Age),
         SibSp = as.numeric(SibSp),
         Parch = as.numeric(Parch),
         Fare = as.numeric(Fare),
         Embarked = as.factor(Embarked)
  )


# Clasificadores

set.seed(100)

#Normalizamos los datos o escalados
#datos_fn[, c(2:20)] <- scale(datos_fn[, c(2:20)])

### Train y Test

library(caTools)
split <- sample.split(train$Survived, SplitRatio = 0.70)

train_fn <- subset(train, split == TRUE)  ## 70% de los datos
test_fn <- subset(train, split == FALSE) ## 30% de los datos






### BOoxting


train_bx1 <-train_fn


matrix_n <- train_bx1%>%
  mutate(Survived = as.numeric(Survived),
         Pclass = as.numeric(Pclass),
         Sex = as.numeric(Sex),
         Age = as.numeric(Age),
         SibSp = as.numeric(SibSp),
         Parch = as.numeric(Parch),
         Fare = as.numeric(Fare),
         Embarked = as.numeric(Embarked))

matrix_n <- as.matrix(matrix_n)

library(xgboost)

# Using the cross validation to estimate our error rate:
param <- list("objective" = "binary:logistic")

cv.nround <- 15
cv.nfold <- 3

xgboost_cv = xgb.cv(param = param,
                    data = matrix_n[, -c(1)],
                    label = matrix_n[, c(1)],
                    nfold = cv.nfold,
                    nrounds = cv.nround)



nround  = 300
fit_xgboost <- xgboost(param =param, data = matrix_n[, -c(1)], label = matrix_n[, c(1)], nrounds=nround)




# Get the feature real names
names <- dimnames(matrix_n)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = fit_xgboost)

# Plotting
xgb.plot.importance(importance_matrix)


# Prediction on test and train sets



## matrix para test y predecir
matrix_test <- test_fn%>%
  mutate(Survived = as.numeric(Survived),
         Pclass = as.numeric(Pclass),
         Sex = as.numeric(Sex),
         Age = as.numeric(Age),
         SibSp = as.numeric(SibSp),
         Parch = as.numeric(Parch),
         Fare = as.numeric(Fare),
         Embarked = as.numeric(Embarked))

matrix_test <- as.matrix(matrix_test)




## prediciendo con los datos de test(25%) y train(75%)
pred_xgboost_test <- predict(fit_xgboost, matrix_test[, -c(1)])
pred_xgboost_train <- predict(fit_xgboost, matrix_n[, -c(1)])

# Since xgboost gives a survival probability prediction, we need to find the best cut-off:
proportion <- sapply(seq(.3,.7,.01),function(step) c(step,sum(ifelse(pred_xgboost_train<step,0,1)!=train[, c(1)])))
dim(proportion)


# Applying the best cut-off on the train set prediction for score checking
predict_xgboost_train <- ifelse(pred_xgboost_train<proportion[,which.min(proportion[2,])][1],0,1)
head(predict_xgboost_train)
score <- sum(train[, c(1)] == predict_xgboost_train)/nrow(train)
score




# Applying the best cut-off on the test set
predict_xgboost_test <- ifelse(pred_xgboost_test<proportion[,which.min(proportion[2,])][1],0,1)
test <- as.data.frame(test) # Conveting the matrix into a dataframe



##test
score <- sum(matrix_test[, c(1)] == predict_xgboost_test)/nrow(matrix_test)
score





