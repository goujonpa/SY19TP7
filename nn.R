# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

library(nnet) # Neural networks
library(e1071) # tune
library(pROC)  # ROC
library(caret) # for the createfolds for 6-fold CV
library(xlsx) # Easy xls export 


nn_analysis = function(X, y, filename="", main="", trace=F) {
    # workig dataset
    df = cbind(as.data.frame(X), y)
    colnames(df)[ncol(df)]="y"
    # xls export name
    XLS = paste("./csv/nn/nn_", filename, "_init.xlsx", sep="")
    # outputs
    l = list()
    l$filename = filename
    # optimisation parameters
    l$DECAY = c(1, 5, 10, 25, 50, 75, 100)/100
    l$SIZE = c(1:10, 15, 20, 25)
    # k folds
    folds = createFolds(y, k=6)
    # performances DF
    perf = data.frame(
        size=integer(),
        decay=double(),
        error=double(),
        sd=double(),
        stringsAsFactors = F
    )
    # test errors vector
    errors = vector(length=6)
    # optimisation
    for (decay in l$DECAY) {
        for (size in l$SIZE) {
            for (i in 1:6) {
                # fit
                model = nnet(
                    as.factor(y)~.,
                    data=df[-folds[[i]],],
                    size=size,
                    decay=decay,
                    trace=trace
                )
                # pred
                pred = predict(model, newdata=df[folds[[i]],], type="class")
                # measure error
                errors[i] = length(which(pred != df[folds[[i]],]$y))/length(df[folds[[i]],]$y)
            }
            # keep perf
            perf[nrow(perf)+1,] = c(size, decay, mean(errors), sd(errors))
        }
    }
    # saving perf
    l$bestpar = perf[which(perf$error == min(perf$error)),]
    l$bestperf = l$bestpar$error
    l$perf = perf
    write.xlsx(l$bestpar, XLS, sheetName="Best Par.")
    # if several best parameters, keep one
    if (nrow(l$bestpar) > 1) {
        l$selpar = l$bestpar[with(l$bestpar, order(sd, -size, -decay)),][1,]
    } else {
        l$selpar = l$bestpar
    }
    write.xlsx(l$selpar, XLS, sheetName="Sel. Par.", append=T)
    write.xlsx(l$bestperf, XLS, sheetName="Best Perf.", append=T)
    write.xlsx(l$perf, XLS, sheetName="Perf.", append=T)
    # plot the tuning perfs
    pdf(paste("./plots/nn/nn_", filename, "_init.pdf", sep=""))
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
            perf[which(perf$size == l$SIZE[i]),]$error,
            labels=as.character(l$SIZE[i]),
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
    return (l)
}

nn_decay_opt = function(X, y, size, filename="", main="", trace=F) {
    # work df
    df = cbind(as.data.frame(X), y)
    colnames(df)[ncol(df)]="y"
    # xls export name
    XLS = paste("./csv/nn/nn_", filename, "_decayopt.xlsx", sep="")
    # outputs
    l = list()
    l$filename = filename
    # optimisations parameters
    l$DECAY = c(1:150)/100
    # k folds
    folds = createFolds(y, k=6)
    # perf DF
    perf = data.frame(
        size=integer(),
        decay=double(),
        error=double(),
        sd=double(),
        stringsAsFactors = F
    )
    # test errors vector
    errors = vector(length=6)
    # 6-folds CV on decay optimisation
    for (decay in l$DECAY) {
        for (i in 1:6) {
            model = nnet(
                as.factor(y)~.,
                data=df[-folds[[i]],],
                size=size,
                decay=decay,
                trace=trace
            )
            pred = predict(model, newdata=df[folds[[i]],], type="class")
            errors[i] = length(which(pred != df[folds[[i]],]$y))/length(df[folds[[i]],]$y)
        }
        perf[nrow(perf)+1,] = c(size, decay, mean(errors), sd(errors))
    }
    # saving perfs
    l$bestpar = perf[which(perf$error == min(perf$error)),]
    l$bestperf = l$bestpar$error
    l$perf = perf
    write.xlsx(l$bestpar, XLS, sheetName="Best Par.")
    if (nrow(l$bestpar) > 1) {
        l$selpar = l$bestpar[with(l$bestpar, order(sd, -size, -decay)),][1,]
    } else {
        l$selpar = l$bestpar
    }
    write.xlsx(l$selpar, XLS, sheetName="Sel. Par.", append=T)
    write.xlsx(l$bestperf, XLS, sheetName="Best Perf.", append=T)
    write.xlsx(l$perf, XLS, sheetName="Perf.", append=T)
    # plot the decay optimisation
    pdf(paste("./plots/nn/nn_", filename, "_decayopt.pdf", sep=""))
    plot(
        l$DECAY,
        perf$error,
        xlab="Decay parameter value",
        ylab="Test error estimate",
        main=paste(main, " : Decay parameter optimisation", sep=""),
        type="l"
    )
    dev.off()
    return (l)
}

nn_final_analysis = function(X, y, size, decay, filename="", main="", trace=F, FDA=F) {
    if (FDA) {filename = paste(filename, "_withFDA", sep="")}
    # working DF
    df = cbind(as.data.frame(X), y)
    colnames(df)[ncol(df)]="y"
    # k folds
    folds = createFolds(y, k=6)
    # xls export name
    XLS = paste("./csv/nn/nn_", filename, "_final.xlsx", sep="")
    # outputs
    l = list()
    l$filename = filename
    # conf mat, tst err
    conf_matrix = matrix(rep(0, 6*6), nrow=6, ncol=6)
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
        model = nnet(
            as.factor(y)~.,
            data=train,
            size=size,
            decay=decay,
            trace=trace
        )
        preds = predict(model, newdata=test, type="class")
        conf_matrix = conf_matrix + table(test$y, preds)
        tst_errors[k] = length(which(test$y != preds))/length(test$y)
    }
    # saving perf
    l$mean_tst_err = mean(tst_errors)
    l$sd_tst_err = sd(tst_errors)
    write.xlsx(l, XLS, sheetName="Tst Err.")
    l$cm = as.data.frame.matrix(conf_matrix)
    write.xlsx(l$cm, XLS, sheetName="Conf. Mat.", append=T)
    # boxplot the error rate
    pdf(paste("./plots/nn/nn_", filename, "_errrate.pdf", sep=""))
    boxplot(
        tst_errors,
        ylab="Test error estimate",
        main=paste(
            main,
            " : NN error rate (mean = ",
            round(l$mean_tst_err, digits=3),
            " )",
            sep=""
        ),
        col=colors()[60] # hop les ptites couleurs
    )
    dev.off()
    # final report output
    l$fr = matrix(nrow=1, ncol=6)
    l$fr[1,] = tst_errors
    rownames(l$fr) = paste(c("NN_"), filename, sep="")
    return (l)
}
