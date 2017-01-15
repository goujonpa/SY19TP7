# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

library(MASS)
library(xlsx) # Easy xls export 


fa = function(Xtrain, ytrain, Xtest=NULL, ytest=NULL, filename="", main="") {
    l = list()
    df = cbind(as.data.frame(Xtrain), ytrain)
    colnames(df)[dim(df)[2]] = "y"
    # lda exec
    l$lda = lda(as.factor(y)~., data=df)
    l$scaling = l$lda$scaling
    # composing train df
    l$Xtrain = as.matrix(Xtrain) %*% l$scaling
    l$Xtrain.scaled = scale(l$Xtrain)
    l$train.df = cbind(as.data.frame(l$Xtrain), ytrain)
    colnames(l$train.df)[dim(l$train.df)[2]] = "y"
    l$train.df.scaled = cbind(as.data.frame(l$Xtrain.scaled), ytrain)
    colnames(l$train.df.scaled)[dim(l$train.df.scaled)[2]] = "y"
    
    # Plot over the first factorial plane
    pdf(paste("./plots/fa/fa_", filename, "_trainffp.pdf", sep=""))
    plot(
        l$Xtrain.scaled[,1],
        l$Xtrain.scaled[,2],
        xlab="X1",
        ylab="X2",
        main=paste(main, ": FA result on the first factorial plane (train)"),
        col=y
    )
    dev.off()
    
    if (!is.null(ytest)) {
        # composing test df
        l$Xtest = as.matrix(Xtest) %*% l$scaling
        l$Xtest.scaled = scale(l$Xtest)
        l$test.df = cbind(as.data.frame(l$Xtest), ytest)
        colnames(l$test.df)[dim(l$test.df)[2]] = "y"
        l$test.df.scaled = cbind(as.data.frame(l$Xtest.scaled), ytest)
        colnames(l$test.df.scaled)[dim(l$test.df.scaled)[2]] = "y"
        
        # Plot over the first factorial plane
        pdf(paste("./plots/fa/fa_", filename, "_testffp.pdf", sep=""))
        plot(
            l$Xtest.scaled[,1],
            l$Xtest.scaled[,2],
            xlab="X1",
            ylab="X2",
            main=paste(main, ": FA result on the first factorial plane (test)"),
            col=y
        )
        dev.off()
    }
    return (l)
}
