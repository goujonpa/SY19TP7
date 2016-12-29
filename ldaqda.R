# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

# k folds à l'ancienne, avant l'utilisation massive de tune
for (k in 1:6) {
    # create the folds
    prcf.train = prcf[-folds[[k]],]
    Xpsf.train = Xpsf[-folds[[k]],]
    prcf.test = prcf[folds[[k]],]
    Xpsf.test = Xpsf[folds[[k]]]
}



# Let's try LDA on the principal components !
prcf.lda.model = lda(as.factor(y)~., data=prcf)
prcf.lda.pred = predict(prcf.lda.model, newdata=prcf)
# about 13% training error
length(which(prcf.lda.pred$class != y)) / length(y)


# now why not trying the qda too mdrrrr
prcf.qda.model = qda(as.factor(y)~., data=prcf)
prcf.qda.pred = predict(prcf.qda.model, newdata=prcf)
# about 5% training error
length(which(prcf.qda.pred$class != y)) / length(y)
