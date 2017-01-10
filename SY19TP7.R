# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition
rm(list=ls())

# >>>>> LIBS
library(ggplot2) # Nice plots
library(xlsx) # Easy xls export 
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

# >>>>> DIMENSION REDUCTION
# Two dimension reduction methods : 
# - PCA : Principal component analysis
# - FA : Factor Analysis
# Our experimental protocol will consist in running our 
# machine learning model fitting on :
# 1 - The PC15
# 2 - The PC30
# 3 - (The PC50)
# 4 - The Pure FA
# 5 - The FA over PCA
# >>> PCA :
source("./pc.R")
# execute a pca on the standardized pre-treated individuals
d$pc = pca(d$clean.scaled, d$y)
d$pc15 = d$pc$x[,1:15]
d$pc25 = d$pc$x[,1:25]
d$pc50 = d$pc$x[,1:50]
d$pc100 = d$pc$x[,1:100]
d$pc200 = d$pc$x[,1:200]
# >>> Factor Analysis
source("./fa.R")
fa1 = fa(d$clean, d$y, filename="rawds", main="Raw")
d$clean.fda = fa1$x
fa2 = fa(d$pc200, d$y, filename="pc200", main="PC200")
d$pc200.fda = fa2$x
d$pc200.fda.scaled = fa2$x.scaled

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
l = ldaqda_analysis(d$pc15, y, filename="pc15", main="PC15")
l = ldaqda_analysis(d$pc25, y, filename="pc25", main="PC25")
l = ldaqda_analysis(d$clean.fda, y, filename="clfda", main="Raw FDA")
l = ldaqda_analysis(d$pc200.fda.scaled, y, filename="pc200fda", main="PC200 FDA")

# >>>>> SVM
source("./svm.R")
# first analysis
r = svm_analysis(prc, y, filename="prcomp", main="Pr. Comp.")
# after first analysis we usually get either a linear model with cost = 6
# or a sigmoid model with cost = 1
# so we try to optimise both
b1 = svm_sigmoid_analysis(prc, y, filename="prcomp", main="Pr. Comp.")
cm1 = svm_conf_matrix(prc, y, gamma=b1$gamma, cost=b1$cost, kernel="sigmoid", filename="prcomp", main="Pr. Comp.")
b2 = svm_polynomial_analysis(prc, y, filename="prcomp", main="Pr. Comp.")
cm2 = svm_conf_matrix(prc, y, gamma=b2$gamma, cost=b2$cost, kernel="polynomial", filename="prcomp", main="Pr. Comp.")

r = svm_analysis(prc, y, filename="prcomp25", main="Pr. Comp.")
b1 = svm_sigmoid_analysis(prc25, y, filename="prcomp25", main="Pr. Comp. (25 pca)")
cm1 = svm_conf_matrix(prc25, y, gamma=b1$gamma, cost=b1$cost, kernel="sigmoid", filename="prcomp25", main="Pr. Comp. (25 pca)")
b2 = svm_polynomial_analysis(prc25, y, filename="prcomp25", main="Pr. Comp. (25 pca)")
cm2 = svm_conf_matrix(prc25, y, gamma=b2$gamma, cost=b2$cost, kernel="polynomial", filename="prcomp25", main="Pr. Comp. (25 pca)")

r = svm_analysis(prc, y, filename="prcomp50", main="Pr. Comp. (50 pca)")
b1 = svm_sigmoid_analysis(prc50, y, filename="prcomp50", main="Pr. Comp. (50 pca)")
cm1 = svm_conf_matrix(prc50, y, gamma=b1$gamma, cost=b1$cost, kernel="sigmoid", filename="prcomp50", main="Pr. Comp. (50 pca)")
b2 = svm_polynomial_analysis(prc50, y, filename="prcomp50", main="Pr. Comp. (50 pca)")
cm2 = svm_conf_matrix(prc50, y, gamma=b2$gamma, cost=b2$cost, kernel="polynomial", filename="prcomp50", main="Pr. Comp. (50 pca)")

# >>>>> Random Forests
source("./randomf.R")
r = rf_analysis(prc, y, filename="prcomp", main="Pr. Comp.")
cm =  rf_conf_matrix(prc, y, mtry=r$mtry, ntree=r$ntree, filename="prcomp", main="Pr. Comp.")

r = rf_analysis(prc25, y, filename="prcomp25", main="Pr. Comp. (25 pca)")
cm =  rf_conf_matrix(prc25, y, mtry=r$mtry, ntree=r$ntree, filename="prcomp25", main="Pr. Comp. (25 pca)")

r = rf_analysis(prc50, y, filename="prcomp50", main="Pr. Comp. (50 pca)")
cm =  rf_conf_matrix(prc50, y, mtry=r$mtry, ntree=r$ntree, filename="prcomp50", main="Pr. Comp. (50 pca)")

# >>>>> NN
source("./nn.R")
r = nn_analysis(prc, y, filename="prcomp", main="Pr. Comp.")
# keep the size, optimise the decay
d = decay_opt(prc, y, r$size, filename="prcomp", main="Pr. Comp.")
# build the confusion matrix
m = nn_conf_matrix(prc, y, size=d$size, decay=d$decay, filename="prcomp", main="Pr. Comp.")

r = nn_analysis(prc25, y, filename="prcomp25", main="Pr. Comp. (25 pca)")
d = decay_opt(prc25, y, r$size, filename="prcomp25", main="Pr. Comp. (25 pca)")
m = nn_conf_matrix(prc25, y, size=d$size, decay=d$decay, filename="prcomp25", main="Pr. Comp. (25 pca)")

r = nn_analysis(prc50, y, filename="prcomp50", main="Pr. Comp. (50 pca)")
d = decay_opt(prc50, y, r$size, filename="prcomp50", main="Pr. Comp. (50 pca)")
m = nn_conf_matrix(prc50, y, size=d$size, decay=d$decay, filename="prcomp50", main="Pr. Comp. (50 pca)")


# from the book : 2 parameters to optimise
# - Number of hidden layer
# - Weight Decay
# + starting position of the neural net weights
# + neural network architecture
# + scaled input is necessary
# + simple weight decay doesn't satisfy consistency ==> do we care ?
# + invariance properties built into

# NOTE FROM STUDYING THE BOOKS : BOXPLOT THE TEST ERROR MORE OFTEN !

# Comment on NN : These tools are especially effective in problems with a high signal-to-noise ratio and settings where prediction without interpretation is the goal. They are less effective for problems where the goal is to describe the physical pro- cess that generated the data and the roles of individual inputs.
