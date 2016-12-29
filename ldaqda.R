# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

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
