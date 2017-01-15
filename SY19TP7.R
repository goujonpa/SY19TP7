# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition
rm(list=ls())

# >>>>> LIBS
library(ggplot2) # Nice plots
library(xlsx) # Easy xls export 
library(caret) # for k folds
source("./display.R") # custom display func

# >>>>> DATA LOADING : X individuals, y labels
load("./data/data_expressions.rdata")


# >>>>> WORKIN DATA FRAMES
# We'll use a list to store every working dataset
d = list()
d$y = y # labels
d$raw = X # raw individuals
d$clean = X[, -which(X[1,] == 0)] # without useless black pixel
d$raw.scaled = scale(d$raw) # standardized raw x
d$clean.scaled = scale(d$clean) # standardized cleaned x
d$EXPRESSIONS = c("joie", "surprise", "tristesse", "degout", "colere", "peur")

final_res_df = data.frame()

# >>>>> DIMENSION REDUCTION
# Two dimension reduction methods : 
# - PCA : Principal component analysis
# - FA : Factor Analysis
# Our experimental protocol will consist in running our 
# machine learning model fitting on :
# 1 - The PC15
# 2 - The PC25
# 3 - (The PC50)
# 4 - The Pure FA
# 5 - The FA over PCA

# >>>>> PCA :
source("./pc.R")
# execute a pca on the standardized pre-treated individuals
d$pc = pca(d$clean.scaled, d$y)
d$pc5 = d$pc$x[,1:5]
d$pc15 = d$pc$x[,1:15]
d$pc25 = d$pc$x[,1:25]
d$pc50 = d$pc$x[,1:50]
d$pc100 = d$pc$x[,1:100]
d$pc200 = d$pc$x[,1:200]

# >>>>> Factor Analysis
source("./fa.R")
# important to read : 
# http://stats.stackexchange.com/questions/106121/does-it-make-sense-to-combine-pca-and-lda

# FDA on full dataset
d$raw.fda = fa(d$clean, d$y, filename="rawfdafullds", main="Raw FDA (Full DS)")$Xtrain.scaled
d$pc25.fda = fa(d$pc25, d$y, filename="pc25fdafullds", main="PC25 FDA (Full DS)")$Xtrain.scaled
d$pc5.fda = fa(d$pc5, d$y, filename="pc5fdafullds", main="PC5 FDA (Full DS)")$Xtrain.scaled
d$pc200.fda = fa(d$pc200, d$y, filename="pc200fdafullds", main="PC200 FDA (Full DS)")$Xtrain.scaled

# >>>>> FUNNY : EXPRESSIONS MEAN
# mathematical mean of the images so that we get the "mean faces" 
# for an emotion expression
source("./mean_faces.R")
r = mean_faces = exp_mean(d$raw, d$y, d$EXPRESSIONS)

# >>>>> MEAN REPARTITION for each face expression
# a bit useless but was part of our first explaratory analysis so we'll
# let it there ...
source("./mean_repartitions.R")
r = mean_rep(d$clean.scaled, d$y, d$EXPRESSIONS)

# >>>>> LDA / QDA
# Perform LDA and QDA classifications 
source("./ldaqda.R")
# TO DO plot the test repartitions on the first factorial plan
# for each validation set in the 

# LDA and QDA over 15, 25 PC using 6-fold CV WITHOUT FDA
l = ldaqda_analysis(d$clean, d$y, filename="raw", main="Raw", QDA=F)
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
l = ldaqda_analysis(d$pc15, d$y, filename="pc15", main="PC15")
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
l = ldaqda_analysis(d$pc25, d$y, filename="pc25", main="PC25")
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
pdf("./plots/final_boxplots/LDA_without_FDA.pdf")
par(mar=c(3,3,3,1))
boxplot(
    as.matrix(final_res_df), 
    horizontal=T, use.cols=F, las=2,
    main="LDA / QDA without FDA", names=as.character(c(1:6)))
dev.off()
final_res_df = cbind(final_res_df, apply(final_res_df, 1, mean))
final_res_df = cbind(final_res_df, apply(final_res_df, 1, sd))
write.xlsx(final_res_df, "./csv/final_res/LDA_without_FDA.xlsx")
final_res_df = data.frame()

# LDA and QDA over the FDA using full DS to train (scientificly not correct)
l = ldaqda_analysis(d$raw.fda, d$y, filename="rawfdafullds", main="Raw FDA (FullDS)")
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
l = ldaqda_analysis(d$pc5.fda, d$y, filename="pc5fdafullds", main="PC5 FDA (FullDS)")
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
l = ldaqda_analysis(d$pc25.fda, d$y, filename="pc25fdafullds", main="PC25 FDA (FullDS)")
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
l = ldaqda_analysis(d$pc200.fda, d$y, filename="pc200fdafullds", main="PC200 FDA (FullDS)")
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
pdf("./plots/final_boxplots/LDA_with_FDA.pdf")
par(mar=c(3,3,3,1))
boxplot(
    as.matrix(final_res_df), 
    horizontal=T, use.cols=F, las=2,
    main="LDA / QDA with FDA", names=as.character(c(7:14)))
dev.off()
final_res_df = cbind(final_res_df, apply(final_res_df, 1, mean))
final_res_df = cbind(final_res_df, apply(final_res_df, 1, sd))
write.xlsx(final_res_df, "./csv/final_res/LDA_with_FDA.xlsx")
final_res_df = data.frame()


# LDA and QDA over Raw, PC5, PC25, PC100 using 6cv
l = ldaqda_analysis(d$clean, d$y, filename="rawfda", main="Raw FDA", FDA=T)
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
l = ldaqda_analysis(d$pc5, d$y, filename="pc5fda", main="PC5 FDA", FDA=T)
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
l = ldaqda_analysis(d$pc15, d$y, filename="pc15fda", main="PC15 FDA", FDA=T)
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
l = ldaqda_analysis(d$pc25, d$y, filename="pc25fda", main="PC25 FDA", FDA=T)
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
l = ldaqda_analysis(d$pc50, d$y, filename="pc50fda", main="PC50 FDA", FDA=T)
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
l = ldaqda_analysis(d$pc100, d$y, filename="pc100fda", main="PC100 FDA", FDA=T)
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
l = ldaqda_analysis(d$pc200, d$y, filename="pc200fda", main="PC200 FDA", FDA=T)
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
pdf("./plots/final_boxplots/LDA_with_FDA2.pdf")
par(mar=c(3,3,3,1))
boxplot(
    as.matrix(final_res_df), 
    horizontal=T, use.cols=F, las=2,
    main="LDA / QDA with FDA", names=as.character(c(15:28)))
dev.off()
final_res_df = cbind(final_res_df, apply(final_res_df, 1, mean))
final_res_df = cbind(final_res_df, apply(final_res_df, 1, sd))
write.xlsx(final_res_df, "./csv/final_res/LDA_with_FDA2.xlsx")
final_res_df = data.frame()


# >>>>> SVM
source("./svm.R")
# 6CV
# Didn't process on full dataset because of training time yet

# On PC15
r = svm_analysis(d$pc15, d$y, filename="pc15", main="PC15")
r = svm_sigmoid_analysis(d$pc15, d$y, filename="pc15", main="PC15")
l = svm_final_analysis(
    d$pc15, d$y, 
    gamma=r$selpar1$gamma, cost=r$selpar2$cost, kernel="sigmoid",
    filename="pc15", main="PC15")
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
l = svm_final_analysis(
    d$pc15, d$y, 
    gamma=r$selpar1$gamma, cost=r$selpar2$cost, kernel="sigmoid",
    filename="pc15", main="PC15 FDA", FDA=T)
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
r = svm_polynomial_analysis(d$pc15, d$y, filename="clean", main="PC15")
l = svm_final_analysis(
    d$pc15, d$y, 
    gamma=r$selpar1$gamma, cost=r$selpar2$cost, kernel="polynomial",
    filename="pc15", main="PC15")
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
l = svm_final_analysis(
    d$pc15, d$y, 
    gamma=r$selpar1$gamma, cost=r$selpar2$cost, kernel="polynomial",
    filename="pc15", main="PC15 FDA", FDA=T)
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
# On PC25
r = svm_analysis(d$pc25, d$y, filename="pc25", main="PC25")
r = svm_sigmoid_analysis(d$pc25, d$y, filename="pc25", main="PC25")
l = svm_final_analysis(
    d$pc25, d$y, 
    gamma=r$selpar1$gamma, cost=r$selpar2$cost, kernel="sigmoid",
    filename="pc25", main="PC25")
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
l = svm_final_analysis(
    d$pc25, d$y, 
    gamma=r$selpar1$gamma, cost=r$selpar2$cost, kernel="sigmoid",
    filename="pc25", main="PC25 FDA", FDA=T)
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
r = svm_polynomial_analysis(d$pc25, d$y, filename="clean", main="PC25")
l = svm_final_analysis(
    d$pc25, d$y, 
    gamma=r$selpar1$gamma, cost=r$selpar2$cost, kernel="polynomial",
    filename="pc25", main="PC25")
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
l = svm_final_analysis(
    d$pc25, d$y, 
    gamma=r$selpar1$gamma, cost=r$selpar2$cost, kernel="polynomial",
    filename="pc25", main="PC25 FDA", FDA=T)
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
pdf("./plots/final_boxplots/SVM.pdf")
par(mar=c(3,3,3,1))
boxplot(
    as.matrix(final_res_df), 
    horizontal=T, use.cols=F, las=2,
    main="SVM", names=as.character(c(29:36)))
dev.off()
final_res_df = cbind(final_res_df, apply(final_res_df, 1, mean))
final_res_df = cbind(final_res_df, apply(final_res_df, 1, sd))
write.xlsx(final_res_df, "./csv/final_res/SVM.xlsx")
final_res_df = data.frame()



# >>>>> Random Forests
source("./randomf.R")
# On PC15
r = rf_analysis(d$pc15, d$y, filename="pc15", main="PC15")
l =  rf_final_analysis(
    d$pc15, d$y, 
    mtry=r$selpar$mtry, ntree=r$selpar$ntree, 
    filename="pc15", main="PC15")
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
l =  rf_final_analysis(
    d$pc15, d$y, 
    mtry=r$selpar$mtry, ntree=r$selpar$ntree, 
    filename="pc15fda", main="PC15 FDA", FDA=T)
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
# On PC25
r = rf_analysis(d$pc25, d$y, filename="pc25", main="PC25")
l =  rf_final_analysis(
    d$pc25, d$y, 
    mtry=r$selpar$mtry, ntree=r$selpar$ntree, 
    filename="pc25", main="PC25")
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
l =  rf_final_analysis(
    d$pc25, d$y, 
    mtry=r$selpar$mtry, ntree=r$selpar$ntree, 
    filename="pc25fda", main="PC25 FDA", FDA=T)
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
pdf("./plots/final_boxplots/RF.pdf")
par(mar=c(3,3,3,1))
boxplot(
    as.matrix(final_res_df), 
    horizontal=T, use.cols=F, las=2,
    main="Random Forest", names=as.character(c(37:40)))
dev.off()
final_res_df = cbind(final_res_df, apply(final_res_df, 1, mean))
final_res_df = cbind(final_res_df, apply(final_res_df, 1, sd))
write.xlsx(final_res_df, "./csv/final_res/RF.xlsx")
final_res_df = data.frame()



# >>>>> NN
source("./nn.R")
# On PC15
r = nn_analysis(d$pc15, d$y, filename="pc15", main="PC15")
r = nn_decay_opt(d$pc15, d$y, r$selpar$size, filename="pc15", main="PC15")
l = nn_final_analysis(
    d$pc15, d$y, 
    r$selpar$size, r$selpar$decay, 
    filename="pc15", main="PC15")
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
l = nn_final_analysis(
    d$pc15, d$y, 
    r$selpar$size, r$selpar$decay, 
    filename="pc15", main="PC15 FDA",
    FDA=T)
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
# On PC25
r = nn_analysis(d$pc25, d$y, filename="pc25", main="PC15")
r = nn_decay_opt(d$pc25, d$y, r$selpar$size, filename="pc25", main="PC15")
l = nn_final_analysis(
    d$pc25, d$y, 
    r$selpar$size, r$selpar$decay, 
    filename="pc25", main="PC15")
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
l = nn_final_analysis(
    d$pc25, d$y, 
    r$selpar$size, r$selpar$decay, 
    filename="pc25", main="PC15 FDA",
    FDA=T)
final_res_df = rbind(final_res_df, as.data.frame(l$fr))
pdf("./plots/final_boxplots/NN.pdf")
par(mar=c(3,3,3,1))
boxplot(
    as.matrix(final_res_df), 
    horizontal=T, use.cols=F, las=2,
    main="Neural Networks", names=as.character(c(41:44)))
dev.off()
final_res_df = cbind(final_res_df, apply(final_res_df, 1, mean))
final_res_df = cbind(final_res_df, apply(final_res_df, 1, sd))
write.xlsx(final_res_df, "./csv/final_res/NN.xlsx")
final_res_df = data.frame()
