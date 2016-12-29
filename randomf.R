# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

# random forest 

prcf.rf.model = randomForest(as.factor(y)~., data=prcf)
prcf.rf.pred = predict(prcf.rf.model, newdata=prcf)
print(prcf.rf.model)
importance(prcf.rf.model)
# 0% training error
length(which(prcf.rf.pred != y))/length(y)