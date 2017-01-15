# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

# LDA and QDA analysis

library(caret) # for the createfolds for 6-fold CV
library(MASS) # LDA, QDA
library(pROC) # ROC
library(xlsx) # Easy xls export 
source("./fa.R")


ldaqda_analysis = function(X, y, filename="", main="", FDA=F, QDA=T) {
    # filename update
    if (FDA) {filename = paste(filename, "_withFDA", sep="")}
    # initial data
    df = cbind(as.data.frame(X),y)
    colnames(df)[ncol(df)]="y"
    # conf matrix, errors
    lda.conf_matrix = matrix(rep(0, 6*6), nrow=6, ncol=6)
    qda.conf_matrix = matrix(rep(0, 6*6), nrow=6, ncol=6)
    lda.tst_errors = vector()
    qda.tst_errors = vector()
    # outputs
    l=list()
    l$filename = filename
    # k folds
    folds = createFolds(y, k=6)
    for (k in 1:6) {
        train = df[-folds[[k]],]
        test = df[folds[[k]],]
        # FDA
        if (FDA) {
            fda = fa(train[,-dim(train)[2]], train[, dim(train)[2]],
                   test[, -dim(test)[2]], test[, dim(test)[2]],
                   filename=filename, main=main)
            train = fda$train.df.scaled
            test = fda$test.df.scaled
        }
        # LDA
        lda.model = lda(as.factor(y)~., data=train)
        lda.preds = predict(lda.model, newdata=test)
        lda.conf_matrix = lda.conf_matrix + table(test$y, lda.preds$class)
        lda.tst_errors[k] = length(which(test$y != lda.preds$class))/length(test$y)
        # QDA
        if (QDA) {
            qda.model = qda(as.factor(y)~., data=train)
            qda.preds = predict(qda.model, newdata=test)
            qda.conf_matrix = qda.conf_matrix + table(test$y, qda.preds$class)
            qda.tst_errors[k] = length(which(test$y != qda.preds$class))/length(test$y)
        }
    }

    # mean, sd
    l$lda_tst_err = mean(lda.tst_errors)
    l$lda_sd_err = sd(lda.tst_errors)
    l$qda_tst_err = mean(qda.tst_errors)
    l$qda_sd_err = sd(qda.tst_errors)

    # saving the perfs
    xls = paste("./csv/ldaqda/ldaqda_", filename, "_perfs.xlsx", sep="")
    write.xlsx(l, xls, sheetName="Test errors")
    l$lda_cm = as.data.frame.matrix(lda.conf_matrix)
    l$qda_cm = as.data.frame.matrix(qda.conf_matrix)
    write.xlsx(l$lda_cm, xls, sheetName="LDA Conf. Mat.", append=T)
    write.xlsx(l$qda_cm, xls, sheetName="QDA Conf. Mat.", append=T)
    
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
    # final report output
    l$fr = matrix(nrow=2, ncol=6)
    l$fr[1,] = lda.tst_errors
    if (QDA) {l$fr[2,] = qda.tst_errors} else {l$fr[2,] = rep(NA, 6)}
    rownames(l$fr) = paste(c("LDA_", "QDA_"), filename, sep="")
    return (l)
}

