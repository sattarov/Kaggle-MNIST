#try H2O deep learning
setwd("D:\\DOCS\\kaggle\\MNIST")
library(h2o)
library(caret)
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
#zeroCols = unique(c(which(apply(training==0,2,all)), which(apply(cv==0,2,all)), which(apply(testing==0,2,all))))
zeroCols = unique(c(which(apply(training==0,2,all)), which(apply(testing==0,2,all))))
training = training[, -zeroCols]
cv = cv[, -zeroCols]
testing = testing[, -zeroCols]


localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, max_mem_size = '6g',nthreads = 3)
#k = 60
#accuracies_train = NULL
#accuracies_cv = NULL
#target_train = as.h2o(localH2O, as.data.frame(training$target), key = 'target_train')
#target_cv = as.h2o(localH2O, as.data.frame(cv$target), key = 'target_cv')
#impVars = names(vars)
#for (i in 4:20) {
#lowImpVars = match( impVars[(length(vars)-k):length(vars)], names(training) )
training_h2o <- as.h2o(localH2O, training, key = 'train')
cv_h2o = as.h2o(localH2O, cv, key = 'cv')
model <- h2o.deeplearning(x = 1:(ncol(training_h2o)-1),  # column numbers for predictors
                          y = "target",   # column number for label
                          data = training_h2o, # data in H2O format
                          activation = "MaxoutWithDropout", # or 'Tanh'
                          classification = T,
                          input_dropout_ratio = 0.1, # % of inputs dropout
                          hidden_dropout_ratios = c(0.8,0.5,0.3), # % for nodes dropout
                          balance_classes = TRUE, 
                          #variable_importance = T,
                          hidden = c(1024,512,256), # three layers of 50 nodes
                          epochs = 100) # max. no. of epochs
h2o_prediction <- as.data.frame(h2o.predict(model, cv_h2o))
accuracies_cv = c(accuracies_cv, sum(h2o_prediction$predict==cv$target)/nrow(cv))
h2o_prediction <- as.data.frame(h2o.predict(model, training_h2o))
accuracies_train = c(accuracies_train, sum(h2o_prediction$predict==training$target)/nrow(training))
k = k+30
print(k)
print(accuracies_cv)
plot(accuracies_cv)
#}

#trying PCA
training.pca = h2o.prcomp(data = training_h2o[,-ncol(training_h2o)],
                          standardize = T,
                          retx = T)
                          #max_pc = 252)
training.features.pca = h2o.predict(training.pca, training_h2o[,-ncol(training_h2o)], num_pc=252)
training.features.pca[,"target"] = training_h2o$target

cv.pca = h2o.prcomp(data = cv_h2o[,-ncol(cv_h2o)],
                    standardize = T,
                    retx = T)
                    #max_pc = 252)
cv.features.pca = h2o.predict(cv.pca, cv_h2o[,-ncol(cv_h2o)], num_pc=252)
cv.features.pca[,"target"] = cv_h2o$target

#random forest
model = h2o.randomForest(x = 1:(ncol(training_h2o)-1), 
                         y = "target",
                         data = training_h2o, 
                         classification = T, 
                         ntree = 500, 
                         depth = 100, 
                         mtries = 41,
                         verbose = T)

h2o_prediction <- as.data.frame(h2o.predict(model, cv_h2o))
confusionMatrix(data = h2o_prediction$predict, reference = cv$target)
sum(h2o_prediction$predict==cv$target)/nrow(cv)
class(h2o_prediction$predict)
