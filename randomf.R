# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

# random forest

library(randomForest) # RF
library(pROC)  # ROC
library(caret) # for the createfolds for 6-fold CV
library(xlsx) # Easy xls export 


rf_analysis = function(X, y, filename="", main="") {
    # working DF
    df = cbind(as.data.frame(X), y)
    colnames(df)[ncol(df)]="y"
    # outputs
    l = list()
    l$filename = filename
    # xls export name
    XLS = paste("./csv/rf/rf_", filename, "_init.xlsx", sep="")
    l$NTREE = c(10, 50, 100, 250, 500)
    l$MTRY = c(1:ncol(X))
    df.tune = tune(
        randomForest,
        as.factor(y)~.,
        data=df,
        ranges=list(
            ntree=l$NTREE,
            mtry=l$MTRY
        )
    )
    # saving perfs
    l$perf = df.tune$performances
    perf = l$perf
    l$bestpar = df.tune$best.parameters
    l$bestperf = df.tune$best.performance
    write.xlsx(l$bestperf, XLS, sheetName="Best Perf.")
    # sel best par set if > 1
    if (nrow(l$bestpar) > 1) {
        l$selpar = l$bestpar[with(l$bestpar, order(sd, -ntree, mtry)),][1,]
    } else {
        l$selpar = l$bestpar
    }
    write.xlsx(l$selpar, XLS, sheetName="Sel. Par.", append=T)
    write.xlsx(l$bestpar, XLS, sheetName="Best Par.", append=T)
    write.xlsx(l$perf, XLS, sheetName="Perf.", append=T)
    # plot the tuning perfs
    pdf(paste("./plots/rf/rf_", filename, "_init.pdf", sep=""))
    layout(cbind(1,2), widths=c(7,3))
    plot(
        1,
        type="n",
        xlim=c(1, ncol(X)),
        ylim=c(0, 0.8),
        ylab="Test error estimate",
        xlab="mtry parameter value",
        main=paste(main, " RF models performances", sep="")
    )
    for (i in l$NTREE){
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
    return (l)
}

rf_final_analysis = function(X, y, mtry, ntree, filename="", main="", FDA=F) {
    if (FDA) {filename = paste(filename, "_withFDA", sep="")}
    # working DF
    df = cbind(as.data.frame(X), y)
    colnames(df)[ncol(df)]="y"
    # k folds
    folds = createFolds(y, k=6)
    # xls export name
    XLS = paste("./csv/rf/rf_", filename, "_final.xlsx", sep="")
    l=list()
    # a matrix to add the different confusion matrix over time
    conf_matrix = matrix(rep(0, 6*6), nrow=6, ncol=6)
    # a test error vector to store the test errors over time
    tst_errors = vector(length=6)
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
        # fit and predict
        model = randomForest(
            as.factor(y)~.,
            data=train,
            ntree=ntree,
            mtry=max(mtry, (ncol(train)-1))
        )
        preds = predict(model, newdata=test)
        conf_matrix = conf_matrix + table(test$y, preds)
        tst_errors[k] = length(which(test$y != preds))/length(test$y)
    }
    # save perfs
    l$mean_tst_err = mean(tst_errors)
    l$sd_tst_err = sd(tst_errors)
    write.xlsx(l, XLS, sheetName="Tst Err.")
    l$cm = as.data.frame.matrix(conf_matrix)
    write.xlsx(l$cm, XLS, sheetName="Conf. Mat.", append=T)
    # plot the error rates
    pdf(paste("./plots/rf/rf_", filename, "_errrates.pdf", sep=""))
    boxplot(
        tst_errors,
        names=list(
            paste("RF mean: ", round(l$mean_tst_err, digits=3), sep="")
        ),
        ylab="Test error estimate",
        main=paste(
            main,
            " : RF error rates (mean: ",
            round(l$mean_tst_err, digits=3),
            ")",
            sep=""
        ),
        col=colors()[60] # hop les ptites couleurs
    )
    dev.off()
    # final report output
    l$fr = matrix(nrow=1, ncol=6)
    l$fr[1,] = tst_errors
    rownames(l$fr) = paste(c("RF_"), filename, sep="")
    return (l)
}
