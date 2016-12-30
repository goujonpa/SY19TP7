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
library(caret) # for the createfolds for 6-fold CV

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
# Funny : mean face of an expression

for (i in 1:6) {
    pdf(paste("./plots/meanface_", as.character(i), ".pdf", sep=""))
    ys = apply(
        Xf[which(Xf$y == as.character(i)),-3661], 
        2, 
        mean
    )
    disp(ys, 60, 70)
    dev.off()
}

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
# Create 6 folds of 36 individuals
folds = createFolds(y, k=6)

# >>>>> LDA / QDA
# source("./ldaqda.R)

# >>>>> SVM
# source("./svm.R")

# as we are big tunning fans, we are primarily going to tune our model
# in order to choose the model we will use and try to optimise
prcf.svm.tune = tune(
    svm, 
    as.factor(y)~., 
    data=prcf,
    ranges=list(
        kernel=c("linear", "polynomial", "radial", "sigmoid"),
        cost=c(0.0001, 0.001, 0.01, 0.1, 1:10),
        degree=c(1:5)
    )
)
# save performances
write.csv(prcf.svm.tune$performances, file="./csv/prcf_svm_perfomances.csv")

print(prcf.svm.tune$best.model)
print(prcf.svm.tune$best.performance)

# >>>>> PLOT : the tuning performance results
prcf.svm.perf = prcf.svm.tune$performances
pdf("./plots/svm_tune1.pdf")
layout(cbind(1,2), widths=c(7,3))
plot(
    1, 
    type="n", 
    xlim=c(0, 10),
    ylim=c(0, 0.8),
    ylab="Test error estimate",
    xlab="Cost value",
    main="SVM models performances"
)
for (i in 1:5){
    text(
        unique(prcf.svm.perf$cost),
        prcf.svm.perf[which(prcf.svm.perf$kernel == "polynomial" & prcf.svm.perf$degree == i),]$error,
        labels=as.character(i),
        col=colors()[10*i]
    )
}
text(
    unique(prcf.svm.perf$cost),
    prcf.svm.perf[which(prcf.svm.perf$kernel == "linear" & prcf.svm.perf$degree == 1),]$error,
    labels="L",
    col=colors()[60]
)
text(
    unique(prcf.svm.perf$cost),
    prcf.svm.perf[which(prcf.svm.perf$kernel == "radial" & prcf.svm.perf$degree == 1),]$error,
    labels="R",
    col=colors()[70]
)
text(
    unique(prcf.svm.perf$cost),
    prcf.svm.perf[which(prcf.svm.perf$kernel == "sigmoid" & prcf.svm.perf$degree == 1),]$error,
    labels="S",
    col=colors()[80]
)
par(mar=c(0,0,0,0))
plot.new()
leg = c(
    "Polynomial d=1",
    "Polynomial d=2",
    "Polynomial d=3",
    "Polynomial d=4",
    "Polynomial d=5",
    "Linear",
    "Radial",
    "Sigmoid"
)
legend(0, 1, legend=leg, lty=1, col=colors()[(1:8)*10])
dev.off()

# Another plotting test :
dev.off()
ys = apply(
    Xpsf[which(Xpsf$y == "5"),-3661], 
    2, 
    mean
)
plot(
    1:3660, 
    ys
)
sdv = apply(Xpsf[which(Xpsf$y=="6"),-3661], 2, sd)
for (i in 1:3660) {
    segments(i, ys[i]+sdv[i], i, ys[i]-sdv[i])
}




# THEN HERE optimise cost

# THEN HERE CONFUSION MATRIX

# THEN HERE ROC CURVE (if possible) => seems possible, proc => multiclass.roc

# primary analysis of different models using 6-folds CV to evaluate the test error 
    # fit the models
    prcf.svm.model = svm(as.factor(y)~., data=prcf, scale=F)
    prcf.svm.pred = predict(prcf.svm.model, newdata=prcf)
    # 0% training error oklm
    length(which(prcf.svm.pred != y))/length(y)
    
    Xpsf.svm.model = svm(as.factor(y)~., data=Xpsf, scale=F)
    Xpsf.svm.pred = predict(Xpsf.svm.model, newdata=Xpsf)
    # 0.0092.. 
    length(which(Xpsf.svm.pred != y))/length(y)
    
    
}


# >>>>> NN
# source("./nn.R")

# >>>>> Random Forests 
# source("./randomf.R")
