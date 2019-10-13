install.packages('caret')
install.packages('rpart')
install.packages('rpart.plot')
install.packages('MLmetrics')
install.packages('pROC')
install.packages('data.table', dependencies = TRUE)


library(data.table)
library(caret)
library(rpart)
library(rpart.plot)
library(MLmetrics)
library(ggplot2)

#get working directory
getwd()

#setting seed
set.seed(123)

#Load dataset save into df
df <- read.csv('/Users/jamesvalles/Desktop/DATASCIENCE/homework2/HW2_steam.csv')

#Getting a summary of the data
summary(df)

#create a training set and a test set
inTrain <- createDataPartition(y = df$score, p = 0.67, list = FALSE)

#select training rows
training <- df[inTrain,]

#select testing rows
testing <- df[-inTrain,]

#Build a decision tree (model)
tree <- rpart(score ~ ., data = training, method = 'class',)

#Plot tree answer for Part 1a
prp(tree)

##determining the relative variable importance, answer for Part 1b
tree$variable.importance

#create a table to show error rate and complexity of decision tree, answer for 1c
printcp(tree)

#create graph to show error rate and complexity of decision tree, answer for 1c
plotcp(tree)

#predict class labels using test data, answer for 1d
tree.pred <- predict(tree, testing, type = "class")

#summary detailing actual and predicted outcomes in the testing (validation) set, result 1d
confusionMatrix(tree.pred, testing$score)

#prune the tree using the optimal complexity parameter 0.020440
pruned <- prune(tree, cp=0.020440)

#generate pruned tree image
prp(pruned)

#predict the class labels for test data, using pruned tree 
tree.pruned = predict(pruned, testing, type = "class")

#summary detailing actual and predicted outcomes in the testing (validation) set using pruned model
confusionMatrix(tree.pruned, testing$score)

#knn needs to work with numeric variables, we have several factors (true/false) in dataframe, so will create dummy variable
dummies <- dummyVars(score ~ ., data = df)
df_dummies = as.data.frame(predict(dummies, newdata = df))
df_dummies$score = df$score

#create new training and test split
inTrain  <- createDataPartition(y=df_dummies$score, p = 0.7, list = FALSE)
training <- df_dummies[inTrain,]
testing <- df_dummies[-inTrain,]

#scale and center our variables
preprocess <- preProcess(training, method = c("center", "scale"))
train_tranformed <- predict(preprocess, training)
test_transformed <- predict(preprocess, testing)

#specify training hyperparameters, use 5-fold cross-validation with 5 repeats
fitControl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)

#fit model to training data
knn1 <- train(score ~., data= train_tranformed, method = "knn", trControl = fitControl)

#estimate most important variables (relative importance), part 2a
varImp(knn1)

#plot accurary vs number of neighboring points (neighbors k)
plot(knn1)

#predict class label for the test data
pred <- predict(knn1, newdata = test_transformed)

#results of how well the model performed
confusionMatrix(pred, testing$score)


