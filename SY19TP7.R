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
# on PC15
r = svm_analysis(d$pc15, d$y, filename="pc15", main="PC15")
r = svm_sigmoid_analysis(d$pc15, d$y, filename="pc15", main="PC15")
r = svm_final_analysis(
    d$pc15, d$y, 
    gamma=r$selpar1$gamma, 
    cost=r$selpar2$cost, 
    kernel="sigmoid",
    filename="pc15", main="PC15")
r = svm_polynomial_analysis(d$pc15, d$y, filename="pc15", main="PC15")
r = svm_final_analysis(
    d$pc15, d$y, 
    gamma=r$selpar1$gamma, 
    cost=r$selpar2$cost, 
    kernel="polynomial",
    filename="pc15", main="PC15")
# on PC25
r = svm_analysis(d$pc25, d$y, filename="pc25", main="PC25")
r = svm_sigmoid_analysis(d$pc25, d$y, filename="pc25", main="PC25")
r = svm_final_analysis(
    d$pc25, d$y, 
    gamma=r$selpar1$gamma, 
    cost=r$selpar2$cost, 
    kernel="sigmoid",
    filename="pc25", main="PC25")
r = svm_polynomial_analysis(d$pc25, d$y, filename="pc25", main="PC25")
r = svm_final_analysis(
    d$pc25, d$y, 
    gamma=r$selpar1$gamma, 
    cost=r$selpar2$cost, 
    kernel="polynomial",
    filename="pc25", main="PC25")
# on PC50 : not yet

# on raw fda :
r = svm_analysis(d$clean.fda , d$y, filename="rawfda", main="Raw FDA")
r = svm_sigmoid_analysis(d$clean.fda, d$y, filename="rawfda", main="Raw FDA")
r = svm_final_analysis(
    d$clean.fda, d$y, 
    gamma=r$selpar1$gamma, 
    cost=r$selpar2$cost, 
    kernel="sigmoid",
    filename="rawfda", main="Raw FDA")
r = svm_polynomial_analysis(d$clean.fda, d$y, filename="rawfda", main="Raw FDA")
r = svm_final_analysis(
    d$clean.fda, d$y, 
    gamma=r$selpar1$gamma, 
    cost=r$selpar2$cost, 
    kernel="polynomial",
    filename="rawfda", main="Raw FDA")
# on PCA + FDA
r = svm_analysis(d$pc200.fda.scaled , d$y, filename="pc200fda", main="PC200 FDA")
r = svm_sigmoid_analysis(d$pc200.fda.scaled, d$y, filename="pc200fda", main="PC200 FDA")
r = svm_final_analysis(
    d$pc200.fda.scaled, d$y, 
    gamma=r$selpar1$gamma, 
    cost=r$selpar2$cost, 
    kernel="sigmoid",
    filename="pc200fda", main="PC200 FDA")
r = svm_polynomial_analysis(d$pc200.fda.scaled, d$y, filename="pc200fda", main="PC200 FDA")
r = svm_final_analysis(
    d$pc200.fda.scaled, d$y, 
    gamma=r$selpar1$gamma, 
    cost=r$selpar2$cost, 
    kernel="polynomial",
    filename="pc200fda", main="PC200 FDA")


# >>>>> Random Forests
source("./randomf.R")
# On PC15
r = rf_analysis(d$pc15, d$y, filename="pc15", main="PC15")
r =  rf_final_analysis(d$pc15, d$y, mtry=r$selpar$mtry, ntree=r$selpar$ntree, filename="pc15", main="PC15")
# On PC25
r = rf_analysis(d$pc25, d$y, filename="pc25", main="PC25")
r =  rf_final_analysis(d$pc25, d$y, mtry=r$selpar$mtry, ntree=r$selpar$ntree, filename="pc25", main="PC25")
# On raw FDA
r = rf_analysis(d$clean.fda, d$y, filename="rawfda", main="Raw FDA")
r =  rf_final_analysis(d$clean.fda, d$y, mtry=r$selpar$mtry, ntree=r$selpar$ntree, filename="rawfda", main="Raw FDA")
# On PC + FDA
r = rf_analysis(d$pc200.fda.scaled, d$y, filename="pc200fda", main="PC200 FDA")
r =  rf_final_analysis(d$pc200.fda.scaled, d$y, mtry=r$selpar$mtry, ntree=r$selpar$ntree, filename="pc200fda", main="PC200 FDA")

# >>>>> NN
source("./nn.R")
# On PC15
r = nn_analysis(d$pc15, d$y, filename="pc15", main="PC15")
r = nn_decay_opt(d$pc15, d$y, r$selpar$size, filename="pc15", main="PC15")
r = nn_final_analysis(d$pc15, d$y, r$selpar$size, r$selpar$decay, filename="pc15", main="PC15")
# On PC25
r = nn_analysis(d$pc25, d$y, filename="pc25", main="PC25")
r = nn_decay_opt(d$pc25, d$y, r$selpar$size, filename="pc25", main="PC25")
r = nn_final_analysis(d$pc25, d$y, r$selpar$size, r$selpar$decay, filename="pc25", main="PC25")
# On raw fda
r = nn_analysis(d$clean.fda, d$y, filename="rawfda", main="Raw FDA", trace=T)
r = nn_decay_opt(d$clean.fda, d$y, r$selpar$size, filename="rawfda", main="Raw FDA", trace=T)
r = nn_final_analysis(d$clean.fda, d$y, r$selpar$size, r$selpar$decay, filename="rawfda", main="Raw FDA")
# On PC + FDA
r = nn_analysis(d$pc200.fda.scaled, d$y, filename="pc200fda", main="PC200 FDA")
r = nn_decay_opt(d$pc200.fda.scaled, d$y, r$selpar$size, filename="pc200fda", main="PC200 FDA")
r = nn_final_analysis(d$pc200.fda.scaled, d$y, r$selpar$size, r$selpar$decay, filename="pc200fda", main="PC200 FDA")



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
