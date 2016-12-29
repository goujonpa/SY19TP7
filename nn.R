# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

# now let's try neural networks
prcf.nn.model = nnet(as.factor(y)~., data=prcf, size=10, linout=T, decay=0.001, maxit=200)
prcf.nn.pred = predict(prcf.nn.model, newdata=prc, type="class")
# about 3% training error
length(which(prcf.nn.pred != y))/length(y)
