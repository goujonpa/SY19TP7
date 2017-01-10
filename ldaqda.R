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
    colnames(df)[ncol(df)]="y"

    # >>>>> 6-folds cross validation
    # As we use the tune method for the next models optimisations
    # we will first use the old school manual k-folds cross validation
    # Why 6-folds ? 'cause 216/6 = 36, round number
    folds = createFolds(y, k=6)

    # a matrix to add the different confusion matrix over time
    lda.conf_matrix = matrix(rep(0, 6*6), nrow=6, ncol=6)
    qda.conf_matrix = matrix(rep(0, 6*6), nrow=6, ncol=6)
    # a test error vector to store the test errors over time
    lda.tst_errors = vector(length=6)
    qda.tst_errors = vector(length=6)

    for (k in 1:6) {
        # LDA : fit and predict
        lda.model = lda(as.factor(y)~., data=df[-folds[[k]],])
        lda.preds = predict(lda.model, newdata=df[folds[[k]],])

        # QDA : fit and predict
        qda.model = qda(as.factor(y)~., data=df[-folds[[k]],])
        qda.preds = predict(qda.model, newdata=df[folds[[k]],])

        # build the confusion matrix
        lda.conf_matrix = lda.conf_matrix + table(df[folds[[k]],]$y, lda.preds$class)
        qda.conf_matrix = qda.conf_matrix + table(df[folds[[k]],]$y, qda.preds$class)

        # measure the test error
        lda.tst_errors[k] = length(which(df[folds[[k]],]$y != lda.preds$class))/length(df[folds[[k]],]$y)
        qda.tst_errors[k] = length(which(df[folds[[k]],]$y != qda.preds$class))/length(df[folds[[k]],]$y)
    }

    # mean, sd
    l=list()
    l$lda_tst_err = mean(lda.tst_errors)
    l$lda_sd_err = sd(lda.tst_errors)
    l$qda_tst_err = mean(qda.tst_errors)
    l$qda_sd_err = sd(qda.tst_errors)

    # saving the perfs
    xls = paste("./csv/ldaqda/ldaqda_", filename, "_perfs.xlsx", sep="")
    write.xlsx(l, xls, sheetName="Test errors")
    write.xlsx(as.data.frame.matrix(lda.conf_matrix), xls, sheetName="LDA Conf. Mat.", append=T)
    write.xlsx(as.data.frame.matrix(qda.conf_matrix), xls, sheetName="QDA Conf. Mat.", append=T)
    
    # append conf matrix to l
    l$lda_cm = as.data.frame.matrix(lda.conf_matrix)
    l$qda_cm = as.data.frame.matrix(qda.conf_matrix)

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

