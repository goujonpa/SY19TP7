# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

# LDA and QDA analysis

library(caret) # for the createfolds for 6-fold CV
library(MASS) # LDA, QDA
library(pROC) # ROC

ldaqda_analysis = function(X, y, filename="", main="") {
    
    # >>>>> DATA FRAME
    df = cbind(as.data.frame(X),y)
    
    # >>>>> 6-folds cross validation
    # As we use the tune method for the next models optimisations
    # we will first use the old school manual k-folds cross validation
    # Why 6-folds ? 'cause 216/6 = 36, round number
    folds = createFolds(y, k=6)
    
    # an array to save the different confusion matrix over time
    lda.confs = array(dim=c(6, 6, 6))
    qda.confs = array(dim=c(6, 6, 6))
    # a matrix to add the different confusion matrix over time
    lda.conf_matrix = matrix(rep(0, 6*6), nrow=6, ncol=6)
    qda.conf_matrix = matrix(rep(0, 6*6), nrow=6, ncol=6)
    # a test error vector to store the test errors over time
    lda.tst_errors = vector(length=6)
    qda.tst_errors = vector(length=6)
    
    for (k in 1:6) {
        # split into folds
        train.df = df[-folds[[k]],]
        test.df = df[folds[[k]],]
        
        # LDA : fit and predict
        lda.model = lda(as.factor(y)~., data=train.df)
        lda.preds = predict(lda.model, newdata=test.df)
        
        # QDA : fit and predict
        qda.model = qda(as.factor(y)~., data=train.df)
        qda.preds = predict(qda.model, newdata=test.df)
        
        # LDA : build the confusion matrix
        lda.confs[,,k] = table(test.df$y, lda.preds$class)
        lda.conf_matrix = lda.conf_matrix + lda.confs[,,k]
        
        # QDA : build the confusion matrix
        qda.confs[,,k] = table(test.df$y, qda.preds$class)
        qda.conf_matrix = qda.conf_matrix + qda.confs[,,k]
        
        # measure the test error
        lda.tst_errors[k] = length(which(test.df$y != lda.preds$class))/length(test.df$y)
        qda.tst_errors[k] = length(which(test.df$y != qda.preds$class))/length(test.df$y)
    }
    
    # mean, sd
    l=list()
    l$lda_tst_err = mean(lda.tst_errors)
    l$lda_sd_err = sd(lda.tst_errors)
    l$qda_tst_err = mean(qda.tst_errors)
    l$qda_sd_err = sd(qda.tst_errors)
    
    # save the confusion matrix
    write.csv(
        lda.conf_matrix, 
        file=paste("./csv/ldaqda/lda_", filename, "_confmat.csv", sep="")
    )
    write.csv(
        qda.conf_matrix, 
        file=paste("./csv/ldaqda/qda_", filename, "_confmat.csv", sep="")
    )
    write.csv(
        l, 
        file=paste("./csv/ldaqda/ldaqda_", filename, "_l.csv", sep="")
    )

    # plot the error rates
    pdf(paste("./plots/ldaqda/ldaqda_", filename, "_errrates.pdf", sep=""))
    boxplot(
        qda.tst_errors, 
        lda.tst_errors, 
        names=list(
            paste("QDA mean: ", round(l$qda_tst_err, digits=3), sep=""), 
            paste("LDA mean: ", round(l$lda_tst_err, digits=3), sep="")
        ), 
        ylab="Test error estimate", 
        main=paste(main, " : LDA/QDA error rates", sep=""),
        col=colors()[c(60,20)] # hop les ptites couleurs
    )
    dev.off()

    return (l)
}

