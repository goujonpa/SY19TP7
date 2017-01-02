# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

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
    bestpar = perf[which(perf$error == min(perf$error)),]
    bestperf = bestpar$error

    # save performances
    write.csv(
        perf, 
        file=paste("./csv/nn/", filename, "_nn_perfomances.csv", sep="")
    )
    
    write.csv(
        bestpar,
        file=paste("./csv/nn/", filename, "_nn_bestpar.csv", sep="")
    )
    
    write.csv(
        bestperf,
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
    
    return (bestpar)
}

decay_opt = function(X, y, size, filename="", main="") {
    # init df
    df = cbind(as.data.frame(X), y)
    
    DECAY = c(1:150)/100
    folds = createFolds(y, k=6)
    
    perf = data.frame(
        size=integer(),
        decay=double(),
        error=double(),
        sd=double(),
        stringsAsFactors = F
    )
    errors = vector(length=6)
    
    # 6-folds CV on decay optimisation
    for (decay in DECAY) {
        for (i in 1:6) {
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
    
    # plot it 
    pdf(paste("./plots/nn/nn_", filename, "_decayopt.pdf", sep=""))
    plot(
        DECAY,
        perf$error,
        xlab="Decay parameter value",
        ylab="Test error estimate",
        main=paste(main, " : Decay parameter optimisation", sep=""),
        type="l"
    )
    dev.off()
    
    # export csv
    write.csv(perf, file=paste("./csv/nn/nn_", filename, "_decayoptperf.csv", sep=""))
    
    # best par export
    bestpar = perf[which(perf$error == min(perf$error)),]
    bestperf = bestpar$error

    write.csv(
        bestpar,
        file=paste("./csv/nn/", filename, "_nn_bestpar2.csv", sep="")
    )
    
    write.csv(
        bestperf,
        file=paste("./csv/nn/", filename, "_nn_bestperf2.csv", sep="")
    )
    
    return (bestpar)
}

nn_conf_matrix = function(X, y, size, decay, filename="", main="") {
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
        # fit and predict
        model = nnet(
            as.factor(y)~., 
            data=df[-folds[[k]],], 
            size=size,
            decay=decay
        )
        preds = predict(model, newdata=df[folds[[k]],], type="class")
        
        # build the confusion matrix
        confs[,,k] = table(df[folds[[k]],]$y, preds)
        conf_matrix = conf_matrix + confs[,,k]
        
        # measure the test error
        tst_errors[k] = length(which(df[folds[[k]],]$y != preds))/length(df[folds[[k]],]$y)
    }

    # save the stats
    write.csv(conf_matrix, file=paste("./csv/nn/conf_matrix_", filename, ".csv", sep=""))
    write.csv(mean(tst_errors), file=paste("./csv/nn/tst_err_", filename, ".csv", sep=""))
    
    # boxplot the error rate
    pdf(paste("./plots/nn/nn_", filename, "_errrate.pdf", sep=""))
    boxplot(
        tst_errors, 
        ylab="Test error estimate", 
        main=paste(
            main, 
            " : NN error rate (mean = ",
            round(mean(tst_errors), digits=3), 
            " )",
            sep=""
        ),
        col=colors()[60] # hop les ptites couleurs
    )
    dev.off()
    
    return (conf_matrix)
}
