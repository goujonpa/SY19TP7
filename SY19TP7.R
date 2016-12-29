# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

# data loading
load("./data/data_expressions.rdata")
# We get X the individuals
# and y the labels

# display function
source("./display.R")

# >>> Libs
library(MASS) # LDA, QDA
library(nnet) # Neural networks
library(ggplot2) # Nice plots
library(e1071) # SVM
library(randomForest) # Random Forest mdr

# >>> First we want to reduce X size by ignoring the black pixels
# Xp is X without the useless pixels (but cannot be displayed)
Xp = X[, -which(X[1,] == 0)]

# >>> Standardize
# xps is the scaled individuals matrix
Xps = scale(Xp)

# >>>>> proper dataframe
Xpsf = cbind(as.data.frame(Xps), y)

# >>> Principal component analysis
# here we source our pcf custom function
source("./prcomp.R")
# and do a principal component analysis and some plotting
# on the scaled data
pc = pcf(Xps)

# Consulting the plots, we decide to select only the 15 first principal component
# We just select the 15 first principal components
prc = as.data.frame(pc$x[,1:15])
# and build a proper dataframe adding the label column
prcf = cbind(prc, y)

# >>>>> NOTE ON CROSS VALIDATION : 
# 216 /5 not round 
# 216 / 6 = 36 
# => WE WILL USE THE 6-FOLDS CROSS VALIDATION

# >>>>> LDA / QDA
# source("./ldaqda.R)

# >>>>> SVM
# source("./svm.R")
prcf.svm.model = svm(as.factor(y)~., data=prcf, scale=F)
prcf.svm.pred = predict(prcf.svm.model, newdata=prcf)
# 0% training error oklm
length(which(prcf.svm.pred != y))/length(y)

Xpsf.svm.model = svm(as.factor(y)~., data=Xpsf, scale=F)
Xpsf.svm.pred = predict(Xpsf.svm.model, newdata=Xpsf)
# 0.0092.. 
length(which(Xpsf.svm.pred != y))/length(y)

# >>>>> NN
# source("./nn.R")

# >>>>> Random Forests 
# source("./randomf.R")
