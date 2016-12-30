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

library(nnet) # Neural networks
library(ggplot2) # Nice plots
library(randomForest) # Random Forest mdr

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

# Consulting the plots, we decide to select only the 15 first principal component
# We just select the 15 first principal components
prc = as.data.frame(pc$x[,1:15])

# >>>>> LDA / QDA
source("./ldaqda.R")
l1 = ldaqda_analysis(prc, y, filename="prcomp", main="Pr. Comp.")

# >>>>> SVM
source("./svm.R")
# first analysis
r1 = svm_analysis(prc, y, filename="prcomp", main="Pr. Comp.")
# after first analysis we usually get either a linear model with cost = 6
# or a sigmoid model with cost = 1
# so we try to optimise both
new_cost = svm_sigmoid_analysis(prc, y, filename="prcomp", main="Pr. Comp.")

# TO DO : linear model optimisation ========================

conf_matrix = svm_conf_matrix(prc, y, new_cost, filename="prcomp", main="Pr. Comp.")

# TO DO : analysis on xps rather than principal components
#svm_analysis(Xps, y, filename="raw", main="Raw")

# >>>>> Random Forests 
# source("./randomf.R")

# >>>>> NN
# source("./nn.R")


