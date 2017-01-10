# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

# data loading : X individuals, y labels
load("./data/data_expressions.rdata")


# display function
source("./display.R")

# >>> Libs

library(ggplot2) # Nice plots

# >>>>> Initial DATA FRAMES
# Xp is X without the useless pixels (but cannot be displayed)
Xp = X[, -which(X[1,] == 0)]

# standardized Xp
Xps = scale(Xp)

# as data frame
Xpsf = cbind(as.data.frame(Xps), y)

# standardized raw X
Xs = scale(X) # nul sur 20

# as data frame raw X
Xf = cbind(as.data.frame(X),y)

# >>>>> INITIAL DATA VIS
EXPRESSIONS = c(
    "joie",
    "surprise",
    "tristesse",
    "degout",
    "colere",
    "peur"
)

# Funny : mean face of an expression
for (i in 1:6) {
    pdf(paste("./plots/meanface_", as.character(i), ".pdf", sep=""))
    ys = apply(
        Xf[which(Xf$y == as.character(i)),-3661],
        2,
        mean
    )
    disp(ys, 60, 70, paste("Image moyenne : ", EXPRESSIONS[i], sep=""))
    dev.off()
}

# useless test :
# plot(1:4200, X[1,], col=y[1])
# for (i in 2:216) {
#     points(1:4200, X[i,], col=y[i])
# }

# mean repartition for each face expression
for (j in 1:6) {
    pdf(paste("./plots/mean_repartition_", as.character(j), ".pdf", sep=""))
    ys = apply(
        Xpsf[which(Xpsf$y == as.character(j)),-3661],
        2,
        mean
    )
    plot(
        1:3660,
        ys,
        col=j,
        main=paste("Répartition moyenne : ", EXPRESSIONS[j], sep=""),
        xlab="Component",
        ylab="Mean Value"
    )
    dev.off()
}

# >>> Principal component analysis
# here we source our pcf custom function
source("./pc.R")
# and do a principal component analysis and some plotting
# on the scaled data
pc = pc_analysis(Xps, y)

# Consulting the plots, we decide to select only the 15, 25, and 50 first principal components
# We can then make comparisons between those and choose the best fitting model

###### 15 PCA - 64% Variance
prc = as.data.frame(pc$x[,1:15])

###### 25 PCA - 74% Variance
prc25 = as.data.frame(pc$x[,1:25])

###### 50 PCA - 85% Variance
prc50 = as.data.frame(pc$x[,1:50])

# >>>>> LDA / QDA
source("./ldaqda.R")
l = ldaqda_analysis(prc, y, filename="prcomp", main="Pr. Comp.")
l2 = ldaqda_analysis(res, y, filename="megatest", main="Top test")

l = ldaqda_analysis(prc25, y, filename="prcomp25", main="Pr. Comp. (25 pca)")

### Remaining too small.
#l = ldaqda_analysis(prc50, y, filename="prcomp50", main="Pr. Comp. (50 pca)")

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
