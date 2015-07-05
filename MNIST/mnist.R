setwd("D:\\DOCS\\kaggle\\MNIST")
library(kernlab)
library(caret)
library(foreach)
library(doParallel)
training = read.csv("train.csv", header = T, na.string=c("","NA"));
testing = read.csv("test.csv", header = T, na.string=c("","NA"));

#move the target label to the end and save it as a factor
target = training[,1]
training = training[,-1]
training[,"target"] = as.factor(target)

#nzv_table = nearZeroVar(training, saveMetrics = T)
nzv_features = nearZeroVar(training)#, freqCut = 30)
training = training[,-nzv_features]
testing = testing[,-nzv_features]

#Check correlated features
M = abs(cor(training[-ncol(training)]))
diag(M) = 0
corrCols = unique(which(M>0.85, arr.ind = TRUE)[,1])
training = training[,-corrCols]
testing = testing[,-corrCols]


#Partition data on training and cross-validation sets
inTrain = createDataPartition(y=training$target, p=0.75, list=FALSE)
cv = training[-inTrain,]
training = training[inTrain,]

# remove the zero columns
zeroCols = unique(c(which(apply(training==0,2,all)), which(apply(cv==0,2,all)), which(apply(testing==0,2,all))))
training = training[, -zeroCols]
cv = cv[, -zeroCols]
testing = testing[, -zeroCols]


#Reducing the dimentionality to 2D and plotting the dataset
prComp = prcomp(training, scale=TRUE)
plot(prComp$x[,1], prComp$x[,2], col=target, type="p", cex=0.5)

preProc = preProcess(training[,-ncol(training)], method = "pca",thresh = 0.99)
training_preProc_PCA = predict(preProc, training[,-81])
plot(training_preProc_PCA[,1], training_preProc_PCA[,2], col=target)


modelInfo = getModelInfo(model="svmRadial")
modelInfo$svmRadial$grid
s=sigest(as.matrix(training[,-ncol(training)]))
warnings()

cl <- makeCluster(3)
registerDoParallel(cl)
trainControl = trainControl(method = "none", number = 3, repeats = 1,  verboseIter = T, allowParallel = T)
grid = expand.grid(size=182)
modelFit = train(target ~., data = training, method="mlp", preProcess=c("scale","center"), trControl = trainControl, tuneGrid = grid, metric = "Accuracy")
stopCluster(cl)

predictions = predict(modelFit, newdata = cv)
confusionMatrix(predictions, cv$target)
sum(predictions==cv$target)/length(predictions)
save(modelFit, file="mlp_repeatedcv_size182")

#Apply the model on a test set
predictions_testing = predict(modelFit, newdata = testing)
predictions_prob_testing = predict(modelFit, newdata = testing, "prob")
write.csv(predictions_prob_testing, file="submission3.csv", quote=F)
head(predictions_prob_testing)


