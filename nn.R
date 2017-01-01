# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

# # now let's try neural networks
# prcf.nn.model = nnet(as.factor(y)~., data=prcf, size=10, linout=T, decay=0.001, maxit=200)
# prcf.nn.pred = predict(prcf.nn.model, newdata=prc, type="class")
# # about 3% training error
# length(which(prcf.nn.pred != y))/length(y)

# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

# random forest 

library(nnet) # Neural networks
library(e1071) # tune
library(pROC)  # ROC
library(caret) # for the createfolds for 6-fold CV

nn_analysis = function(X, y, filename="", main="") {
    # initial dataframe
    df = cbind(as.data.frame(X), y)
    
    # >>>>> FIRST TUNING 
    # to determine which model we are going to use and try to optimise
    # trying to test different values of mtry and ntree
    DECAY = c(1, 5, 10, 25, 50, 75, 100)/100
    SIZE = c(1:10, 15, 20, 25)
    
    folds = createFolds(y, k=6)
    
    perf = data.frame(
        size=integer(),
        decay=double(),
        error=double(),
        sd=double(),
        stringsAsFactors = F
    )
    errors = vector(length=6)
    
    for (decay in DECAY) {
        for (size in SIZE) {
            for (i in 1:6) {
                print(paste("SIZE",size))
                model = nnet(
                    as.factor(y)~., 
                    data=df[-folds[[i]],], 
                    size=size, 
                    decay=decay
                )
                pred = predict(model, newdata=df[folds[[i]],], type="class")
                errors[i] = length(which(pred != df[folds[[i]],]$y))/length(df[folds[[i]],]$y)
            }
            perf[nrow(perf)+1,] = c(size, decay, mean(errors), sd(errors))
        }
    }

    # save performances
    write.csv(
        perf, 
        file=paste("./csv/nn/", filename, "_nn_perfomances.csv", sep="")
    )
    
    write.csv(
        df.tune$best.parameters,
        file=paste("./csv/nn/", filename, "_nn_bestpar.csv", sep="")
    )
    
    write.csv(
        df.tune$best.performance,
        file=paste("./csv/nn/", filename, "_nn_bestperf.csv", sep="")
    )
    
    # plot the tuning perfs
    pdf(paste("./plots/nn/", filename, "_nn_tune1.pdf", sep=""))
    layout(cbind(1,2), widths=c(7,3))
    plot(
        1, 
        type="n", 
        xlim=c(0, 1),
        ylim=c(0, 0.8),
        ylab="Test error estimate",
        xlab="Decay parameter value",
        main=paste(main, " NN models performances", sep="")
    )
    for (i in 1:13){
        text(
            unique(perf$decay),
            perf[which(perf$size == SIZE[i]),]$error,
            labels=as.character(SIZE[i]),
            col=i
        )
    }
    par(mar=c(0,0,0,0))
    plot.new()
    leg = c(
        "size = 1",
        "size = 2",
        "size = 3",
        "size = 4",
        "size = 5",
        "size = 6",
        "size = 7",
        "size = 8",
        "size = 9",
        "size = 10",
        "size = 15",
        "size = 20",
        "size = 25"
    )
    legend(0, 1, legend=leg, lty=1, col=c(1:13))
    dev.off()
    
    return (df.tune$best.parameters)
}

rf_conf_matrix = function(X, y, mtry, ntree, filename="", main="") {
    # Use a 6-fold cross validation method to build a prediction
    # and the associated confusion matrix
    
    # TO BE EXTENDED WITH MORE PARAMETERS THAN JUST COST WHEN WE HAVE TIME
    
    # >>>>> Initial DATA FRAME
    df = cbind(as.data.frame(X), y)
    folds = createFolds(y, k=6)
    
    # an array to save the different confusion matrix over time
    confs = array(dim=c(6, 6, 6))
    # a matrix to add the different confusion matrix over time
    conf_matrix = matrix(rep(0, 6*6), nrow=6, ncol=6)
    # a test error vector to store the test errors over time
    tst_errors = vector(length=6)
    
    for (k in 1:6) {
        # split into folds
        train.df = df[-folds[[k]],]
        test.df = df[folds[[k]],]
        
        # fit and predict
        model = randomForest(
            as.factor(y)~., 
            data=train.df, 
            ntree=ntree,
            mtry=mtry
        )
        preds = predict(model, newdata=test.df)
        
        # build the confusion matrix
        confs[,,k] = table(test.df$y, preds)
        conf_matrix = conf_matrix + confs[,,k]
        
        # measure the test error
        tst_errors[k] = length(which(test.df$y != preds))/length(test.df$y)
    }
    
    # mean it
    tst_err = mean(tst_errors)
    
    # plot the error rates
    pdf(paste("./plots/rf/rf_", filename, "_errrates.pdf", sep=""))
    boxplot(
        tst_errors, 
        names=list(
            paste("RF mean: ", round(tst_err, digits=3), sep="")
        ), 
        ylab="Test error estimate", 
        main=paste(
            main, 
            " : RF error rates (mean: ", 
            round(tst_err, digits=3),
            ")",
            sep=""
        ),
        col=colors()[c(60,20)] # hop les ptites couleurs
    )
    dev.off()    
    
    # save the stats
    write.csv(conf_matrix, file=paste("./csv/rf/conf_matrix_", filename, ".csv", sep=""))
    write.csv(tst_err, file=paste("./csv/rf/tst_err_", filename, ".csv", sep=""))
    
    return (conf_matrix)
}
