# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

# LDA and QDA analysis

library(caret) # for the createfolds for 6-fold CV
library(MASS) # LDA, QDA
library(pROC) # ROC
library(xlsx) # Easy xls export 


ldaqda_analysis = function(X, y, filename="", main="", testData=NULL, testData.y=NULL) {
    df = cbind(as.data.frame(X),y)
    colnames(df)[ncol(df)]="y"
    lda.conf_matrix = matrix(rep(0, 6*6), nrow=6, ncol=6)
    qda.conf_matrix = matrix(rep(0, 6*6), nrow=6, ncol=6)
    lda.tst_errors = vector()
    qda.tst_errors = vector()
    
    if (is.null(testData)) { # 6 fold cross validation
        folds = createFolds(y, k=6)
        for (k in 1:6) {
            lda.model = lda(as.factor(y)~., data=df[-folds[[k]],])
            lda.preds = predict(lda.model, newdata=df[folds[[k]],])
            qda.model = qda(as.factor(y)~., data=df[-folds[[k]],])
            qda.preds = predict(qda.model, newdata=df[folds[[k]],])
            lda.conf_matrix = lda.conf_matrix + table(df[folds[[k]],]$y, lda.preds$class)
            qda.conf_matrix = qda.conf_matrix + table(df[folds[[k]],]$y, qda.preds$class)
            lda.tst_errors[k] = length(which(df[folds[[k]],]$y != lda.preds$class))/length(df[folds[[k]],]$y)
            qda.tst_errors[k] = length(which(df[folds[[k]],]$y != qda.preds$class))/length(df[folds[[k]],]$y)
        }
    } else { # validation set provided
        testData = as.data.frame(testData)
        if (is.null(testData.y)) {
            stop("Please provide validation set y")
        }
        lda.model = lda(as.factor(y)~., data=df)
        lda.preds = predict(lda.model, newdata=testData)
        qda.model = qda(as.factor(y)~., data=df)
        qda.preds = predict(qda.model, newdata=testData)
        lda.conf_matrix = as.data.frame.matrix(table(testData.y, lda.preds$class))
        qda.conf_matrix = as.data.frame.matrix(table(testData.y, qda.preds$class))
        lda.tst_errors[1] = length(which(testData.y != lda.preds$class))/length(testData.y)
        qda.tst_errors[1] = length(which(testData.y != qda.preds$class))/length(testData.y)
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

    return (l)
}

