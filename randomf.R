# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

# random forest

library(randomForest) # RF
library(pROC)  # ROC
library(caret) # for the createfolds for 6-fold CV

rf_analysis = function(X, y, filename="", main="") {
    # initial dataframe
    df = cbind(as.data.frame(X), y)

    # >>>>> FIRST TUNING
    # to determine which model we are going to use and try to optimise
    # trying to test different values of mtry and ntree
    NTREE = c(10, 50, 100, 250, 500)
    df.tune = tune(
        randomForest,
        as.factor(y)~.,
        data=df,
        ranges=list(
            ntree=NTREE,
            mtry=c(1:15)
        )
    )

    perf = df.tune$performances
    bestpar = df.tune$best.parameters
    bestperf = df.tune$best.performance
    write.csv(
        perf,
        file=paste("./csv/rf/", filename, "_rf_perfomances.csv", sep="")
    )
    write.csv(
        bestpar,
        file=paste("./csv/rf/", filename, "_rf_bestpar.csv", sep="")
    )
    write.csv(
        bestperf,
        file=paste("./csv/rf/", filename, "_rf_bestperf.csv", sep="")
    )

    # plot the tuning perfs
    pdf(paste("./plots/rf/", filename, "_rf_tune1.pdf", sep=""))
    layout(cbind(1,2), widths=c(7,3))
    plot(
        1,
        type="n",
        xlim=c(0, 15),
        ylim=c(0, 0.8),
        ylab="Test error estimate",
        xlab="mtry parameter value",
        main=paste(main, " RF models performances", sep="")
    )
    for (i in NTREE){
        text(
            unique(perf$mtry),
            perf[which(perf$ntree == i),]$error,
            labels=as.character(i),
            col=colors()[i]
        )
    }
    par(mar=c(0,0,0,0))
    plot.new()
    leg = c(
        "ntree = 10",
        "ntree = 50",
        "ntree = 100",
        "ntree = 250",
        "ntree = 500"
    )
    legend(0, 1, legend=leg, lty=1, col=colors()[(1:8)*10])
    dev.off()

    return (bestpar)
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
        col=colors()[60] # hop les ptites couleurs
    )
    dev.off()

    # save the stats
    write.csv(conf_matrix, file=paste("./csv/rf/conf_matrix_", filename, ".csv", sep=""))
    write.csv(tst_err, file=paste("./csv/rf/tst_err_", filename, ".csv", sep=""))

    return (conf_matrix)
}



