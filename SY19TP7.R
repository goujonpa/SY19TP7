# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

# data loading
load("./data/data_expressions.rdata")

# display function
source("./display.R")

# >>>>> NEURAL NETWORK

# >>> Libs
library(MASS)
library(nnet)
library(ggplot2)

# >>> First we want to reduce X size by ignoring the black pixels

# Xp is X without the useless pixels (but cannot be displayed)
Xp = X[, -which(X[1,] == 0)]

# >>> Center and reduce

# xps is the scaled individuals matrix
Xps = scale(Xp)


# >>> Then we want to make an ACP on this

# from web sources, prcomp is prefered
# we use the scaled individuals
pc = prcomp(Xps)
pdf("./plots/prcomp_xps.pdf")
plot(pc)
dev.off()
pdf("./plots/prcomp_xps2.pdf")
biplot(pc)
dev.off()

# proportions of variance explained
pc$vars = pc$sdev^2
pc$varsprop = pc$vars / sum(pc$vars)
pc$cumvarsprop = cumsum(pc$varsprop)

pdf("./plots/pca1.pdf")
plot(pc$varsprop, 
     xlab = "Principal Component", 
     ylab="Proportion of variance explained",
     main="PCA - Proportion of variance explained",
     ylim = c(0,0.2), type="h")
dev.off()

pdf("./plots/pca2.pdf")
plot(pc$varsprop[1:50], 
     xlab = "Principal Component", 
     ylab="Proportion of variance explained",
     main="PCA - Proportion of variance explained (PC1 to PC50)",
     ylim = c(0,0.2), type="h")
dev.off()

pdf("./plots/pca3.pdf")
plot(pc$cumvarsprop, 
     xlab = "Principal Component", 
     ylab="Cumulative proportion of variance explained",
     main="PCA - cumulative proportion of variance explained",
     ylim = c(0,1), type="l")
dev.off()


# pc$cumvarsprop
# text(100, pc$cumvarsprop[15], labels = as.character(pc$cumvarsprop[15]))
# abline(v=15)
# abline(h=pc$cumvarsprop[15])

# We just select the 15 first principal components
prc = as.data.frame(pc$x[,1:15])
# and build a proper dataframe adding the label column
prcf = cbind(prc, y)
colnames(prcf)

# Let's try LDA on the principal components !
prcf.lda.model = lda(as.factor(y)~., data=prcf)
prcf.lda.pred = predict(prcf.lda.model, newdata=prcf)
# about 13% training error
length(which(prcf.lda.pred$class != y)) / length(y)


# now why not trying the qda too mdrrrr
prcf.qda.model = qda(as.factor(y)~., data=prcf)
prcf.qda.pred = predict(prcf.qda.model, newdata=prcf)
# about 5% training error
length(which(prcf.qda.pred$class != y)) / length(y)


# now let's try neural networks
prcf.nn.model = nnet(as.factor(y)~., data=prcf, size=10, linout=T, decay=0.001, maxit=200)
prcf.nn.pred = predict(prcf.nn.model, newdata=prc, type="class")
# about 3% training error
length(which(prcf.nn.pred != y))/length(y)

# random forest 
library(randomForest)
prcf.rf.model = randomForest(as.factor(y)~., data=prcf)
prcf.rf.pred = predict(prcf.rf.model, newdata=prcf)
print(prcf.rf.model)
importance(prcf.rf.model)
# 0% training error
length(which(prcf.rf.pred != y))/length(y)

# and finally svm
library(e1071)
prcf.svm.model = svm(as.factor(y)~., data=prcf, scale=F)
prcf.svm.pred = predict(prcf.svm.model, newdata=prcf)
# 0% training error oklm
length(which(prcf.svm.pred != y))/length(y)

