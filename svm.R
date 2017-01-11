# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial recognition

# SVM analysis function

library(e1071) # SVM
library(pROC)  # ROC
library(caret) # for the createfolds for 6-fold CV
library(xlsx) # Easy xls export 


svm_analysis = function(X, y, filename="", main="") {
    df = cbind(as.data.frame(X), y)
    colnames(df)[ncol(df)]="y"
    l = list()
    XLS = paste("./csv/svm/svm_", filename, "initanalysis.xlsx", sep="")
    
    # First general tuning
    df.tune = tune(
        svm,
        as.factor(y)~.,
        data=df,
        ranges=list(
            kernel=c("linear", "polynomial", "radial", "sigmoid"),
            cost=c(0.0001, 0.001, 0.01, 0.1, 1:10),
            degree=c(1:5)
        )
    )

    # saving perf
    l$perf = df.tune$performances
    perf = l$perf
    l$bestperf = df.tune$best.performance
    write.xlsx(l$bestperf, XLS, sheetName="Best Perf.")
    l$bestpar = df.tune$best.parameters
    if (nrow(l$bestpar) > 1) {
        l$selpar = l$bestpar[with(l$bestpar, order(sd, -cost)),][1,]
    } else {
        l$selpar = l$bestpar
    }
    write.xlsx(l$selpar, XLS, sheetName="Sel. Par.", append=T)
    write.xlsx(l$bestpar, XLS, sheetName="Best Par.", append=T)
    write.xlsx(l$perf, XLS, sheetName="Perf.", append=T)

    # plot the tuning perfs
    pdf(paste("./plots/svm/svm_", filename, "_init.pdf", sep=""))
    layout(cbind(1,2), widths=c(7,3))
    plot(
        1,
        type="n",
        xlim=c(0, 10),
        ylim=c(0, 0.8),
        ylab="Test error estimate",
        xlab="Cost value",
        main=paste(main, " SVM models performances", sep="")
    )
    for (i in 1:5){
        text(
            unique(perf$cost),
            perf[which(perf$kernel == "polynomial" & perf$degree == i),]$error,
            labels=as.character(i),
            col=colors()[10*i]
        )
    }
    text(
        unique(perf$cost),
        perf[which(perf$kernel == "linear" & perf$degree == 1),]$error,
        labels="L",
        col=colors()[60]
    )
    text(
        unique(perf$cost),
        perf[which(perf$kernel == "radial" & perf$degree == 1),]$error,
        labels="R",
        col=colors()[70]
    )
    text(
        unique(perf$cost),
        perf[which(perf$kernel == "sigmoid" & perf$degree == 1),]$error,
        labels="S",
        col=colors()[80]
    )
    par(mar=c(0,0,0,0))
    plot.new()
    leg = c(
        "Polynomial d=1",
        "Polynomial d=2",
        "Polynomial d=3",
        "Polynomial d=4",
        "Polynomial d=5",
        "Linear",
        "Radial",
        "Sigmoid"
    )
    legend(0, 1, legend=leg, lty=1, col=colors()[(1:8)*10])
    dev.off()

    return (l)
}

svm_sigmoid_analysis = function(X, y, filename="", main="") {
    df = cbind(as.data.frame(X), y)
    colnames(df)[ncol(df)]="y"
    l = list()
    XLS = paste("./csv/svm/svm_", filename, "_sigmoid.xlsx", sep="")

    # tune parameters
    l$COST = c(5:15)/10
    l$GAMMA = 1/(c(10:30)*10)

    # first, gamma tuning
    df.tune = tune(
        svm,
        as.factor(y)~.,
        data=df,
        ranges=list(cost=l$COST, gamma=l$GAMMA),
        kernel="sigmoid"
    )
    l$bestpar1 = df.tune$best.parameters
    l$bestperf1 = df.tune$best.performance
    write.xlsx(l$bestpar1, XLS, sheetName="Best Par.1")
    if (nrow(l$bestpar1) > 1) {
        l$selpar1 = l$bestpar1[with(l$bestpar1, order(sd, -cost, -gamma)),][1,]
    } else {
        l$selpar1 = l$bestpar1
    }
    write.xlsx(l$selpar1, XLS, sheetName="Sel. Par.1", append=T)
    write.xlsx(l$bestperf1, XLS, sheetName="Best Perf.1", append=T)

    # gamma fixed, cost tuning
    df.tune2 = tune(
        svm,
        as.factor(y)~.,
        data=df,
        ranges=list(cost=l$COST),
        gamma=l$bestpar1$gamma,
        kernel="sigmoid"
    )
    
    # saving perf
    l$perf2 = df.tune2$performances
    perf = l$perf2
    l$bestpar2 = df.tune2$best.parameters
    l$bestperf2 = df.tune2$best.performance
    write.xlsx(l$perf2, XLS, sheetName="Perf.2", append=T)
    if (nrow(l$bestpar2) > 1) {
        l$selpar2 = l$bestpar2[with(l$bestpar2, order(sd, -cost)),][1,]
    } else {
        l$selpar2 = l$bestpar2
    }
    write.xlsx(l$selpar2, XLS, sheetName="Sel. Par.2", append=T)
    write.xlsx(l$bestpar1, XLS, sheetName="Best Par.2", append=T)
    write.xlsx(l$bestperf1, XLS, sheetName="Best Perf.2", append=T)
    
    # plotting the cost optimisation
    pdf(paste("./plots/svm/svm_", filename, "_sigtune.pdf", sep=""))
    plot(perf$cost, perf$error, type="l", ylab="Test error estimate", xlab="Cost", main=paste(main, " : Sig. SVM cost optimisation"))
    dev.off()

    return (l)
}

svm_polynomial_analysis = function(X, y, filename="", main="") {
    df = cbind(as.data.frame(X), y)
    colnames(df)[ncol(df)]="y"
    l = list()
    XLS = paste("./csv/svm/svm_", filename, "_polynomial.xlsx", sep="")
    
    # tune parameters
    l$COST = c(40:120)/10
    l$GAMMA = 1/(c(10:30)*10)

    # cost tuning
    df.tune = tune(
        svm,
        as.factor(y)~.,
        data=df,
        ranges=list(cost=l$COST, gamma=l$GAMMA),
        degree=1,
        kernel="polynomial"
    )
    
    # saving first perfs
    l$bestpar1 = df.tune$best.parameters
    l$bestperf1 = df.tune$best.performance
    write.xlsx(l$bestpar1, XLS, sheetName="Best Par.1")
    if (nrow(l$bestpar1) > 1) {
        l$selpar1 = l$bestpar1[with(l$bestpar1, order(sd, -cost, -gamma)),][1,]
    } else {
        l$selpar1 = l$bestpar1
    }
    write.xlsx(l$selpar1, XLS, sheetName="Sel. Par.1", append=T)
    write.xlsx(l$bestperf1, XLS, sheetName="Best Perf.1", append=T)
    
    df.tune2 = tune(
        svm,
        as.factor(y)~.,
        data=df,
        ranges=list(cost=l$COST),
        gamma=l$bestpar1$gamma,
        degree=1,
        kernel="polynomial"
    )
    # saving perf
    l$perf2 = df.tune2$performances
    perf = l$perf2
    l$bestpar2 = df.tune2$best.parameters
    l$bestperf2 = df.tune2$best.performance
    write.xlsx(l$perf2, XLS, sheetName="Perf.2", append=T)
    if (nrow(l$bestpar2) > 1) {
        l$selpar2 = l$bestpar2[with(l$bestpar2, order(sd, -cost)),][1,]
    } else {
        l$selpar2 = l$bestpar2
    }
    write.xlsx(l$selpar2, XLS, sheetName="Sel. Par.2", append=T)
    write.xlsx(l$bestpar1, XLS, sheetName="Best Par.2", append=T)
    write.xlsx(l$bestperf1, XLS, sheetName="Best Perf.2", append=T)

    # plotting the cost optimisation
    pdf(paste("./plots/svm/svm_", filename, "_poltune.pdf", sep=""))
    plot(perf$cost, perf$error, type="l", ylab="Test error estimate", xlab="Cost", main=paste(main, " : Pol. SVM cost optimisation"))
    dev.off()
    
    return (l)
}


assignclass = function(x) {
    # EXPERIMENTAL : TO BE USED IN AN APPLY( , , FUN )
    # testing to build a function that assigns the most
    # predicted class to an individual
    return (sample(names(which(table(x) == max(table(x)))),1))
}

svm_final_analysis = function(X, y, cost, gamma, kernel, filename="", main="") {
    df = cbind(as.data.frame(X), y)
    colnames(df)[ncol(df)]="y"
    folds = createFolds(y, k=6)
    XLS = paste("./csv/svm/svm_", filename, kernel, "_final.xlsx", sep="")
    l = list()

    # a matrix to add the different confusion matrix over time
    conf_matrix = matrix(rep(0, 6*6), nrow=6, ncol=6)
    # a test error vector to store the test errors over time
    tst_errors = vector(length=6)

    for (k in 1:6) {
        # fit and predict
        model = svm(
            as.factor(y)~.,
            data=df[-folds[[k]],],
            kernel=kernel,
            cost=cost,
            gamma=gamma,
            degree=1
        )
        preds = predict(model, newdata=df[folds[[k]],])

        # build the confusion matrix
        conf_matrix = conf_matrix + table(df[folds[[k]],]$y, preds)

        # measure the test error
        tst_errors[k] = length(which(df[folds[[k]],]$y != preds))/length(df[folds[[k]],]$y)
    }
    # saving perf
    l$tst_err_mean = mean(tst_errors)
    l$tst_err_sd = sd(tst_errors)
    write.xlsx(l, XLS, sheetName="Tst Err.")
    l$cm = as.data.frame.matrix(conf_matrix)
    write.xlsx(l$cm, XLS, sheetName="Conf. Mat.", append=T)
    
    # boxplot the error rate
    pdf(paste("./plots/svm/svm_", filename, kernel, "_errrate.pdf", sep=""))
    boxplot(
        tst_errors,
        ylab="Test error estimate",
        main=paste(
            main,
            " : SVM error rate (mean = ",
            round(mean(tst_errors), digits=3),
            " )",
            sep=""
        ),
        col=colors()[60] # hop les ptites couleurs
    )
    dev.off()

    return (l)
}
