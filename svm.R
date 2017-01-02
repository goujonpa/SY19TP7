# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial recognition

# SVM analysis function


library(e1071) # SVM
library(pROC)  # ROC
library(caret) # for the createfolds for 6-fold CV

svm_analysis = function(X, y, filename="", main="") {
    # initial dataframe
    df = cbind(as.data.frame(X), y)
    
    # >>>>> FIRST TUNING 
    # to determine which model we are going to use and try to optimise
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
    
    perf = df.tune$performances
    # save performances
    write.csv(
        perf, 
        file=paste("./csv/svm/", filename, "_svm_perfomances.csv", sep="")
    )
    
    write.csv(
        df.tune$best.parameters,
        file=paste("./csv/svm/", filename, "_svm_bestpar.csv", sep="")
    )

    write.csv(
        df.tune$best.performance,
        file=paste("./csv/svm/", filename, "_svm_bestperf.csv", sep="")
    )

    # plot the tuning perfs
    pdf(paste("./plots/svm/", filename, "_svm_tune1.pdf", sep=""))
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
    
    return (df.tune$best.parameters)
}

svm_sigmoid_analysis = function(X, y, filename="", main="") {
    # function to be used when the first analysis tells us that 
    # the sigmoid svm model is the best one to use
    
    # >>>>> Initial DATA FRAME
    df = cbind(as.data.frame(X), y)
    
    # tune parameters
    COST = c(5:15)/10
    GAMMA = 1/(c(10:30)*10)
    
    # cost tuning
    df.tune = tune(
        svm, 
        as.factor(y)~., 
        data=df,
        ranges=list(cost=COST, gamma=GAMMA), 
        kernel="sigmoid"
    )
    bestpar = df.tune$best.parameters
    bestperf = df.tune$best.performance
    write.csv(
        bestpar,
        file=paste("./csv/svm/", filename, "_sigmoidsvm_bestpar.csv", sep="")
    )
    write.csv(
        bestperf,
        file=paste("./csv/svm/", filename, "_sigmoidsvm_bestperf.csv", sep="")
    )
    
    df.tune2 = tune(
        svm, 
        as.factor(y)~., 
        data=df,
        ranges=list(cost=COST), 
        gamma=bestpar$gamma,
        kernel="sigmoid"
    )
    perf = df.tune2$performances
    bestpar$cost = df.tune2$best.parameters$cost
    bestperf = df.tune2$best.performance
    pdf(paste("./plots/svm/sigmoid2_", filename, "_tune.pdf", sep=""))
    plot(perf$cost, perf$error, type="l", ylab="Test error estimate", xlab="Cost", main=paste(main, " : SVM cost optimisation"))
    dev.off()
    write.csv(
        bestpar,
        file=paste("./csv/svm/", filename, "_sigmoid2_svm_bestpar.csv", sep="")
    )
    write.csv(
        bestperf,
        file=paste("./csv/svm/", filename, "_sigmoid2_svm_bestperf.csv", sep="")
    )
    
    # TO DO if we got time to waste
    # - coef0 optimisation
        
    return (bestpar)
}

svm_polynomial_analysis = function(X, y, filename="", main="") {
    # function to be used when the first analysis tells us that 
    # the sigmoid svm model is the best one to use
    
    # >>>>> Initial DATA FRAME
    df = cbind(as.data.frame(X), y)
    
    # tune parameters
    COST = c(40:60)/10
    GAMMA = 1/(c(10:30)*10)
    
    # cost tuning
    df.tune = tune(
        svm, 
        as.factor(y)~., 
        data=df,
        ranges=list(cost=COST, gamma=GAMMA), 
        kernel="polynomial"
    )
    bestpar = df.tune$best.parameters
    bestperf = df.tune$best.performance
    write.csv(
        bestpar,
        file=paste("./csv/svm/", filename, "_poly_svm_bestpar.csv", sep="")
    )
    write.csv(
        bestperf,
        file=paste("./csv/svm/", filename, "_poly_svm_bestperf.csv", sep="")
    )
    
    df.tune2 = tune(
        svm, 
        as.factor(y)~., 
        data=df,
        ranges=list(cost=COST), 
        gamma=bestpar$gamma,
        degree=1,
        kernel="polynomial"
    )
    perf = df.tune2$performances
    bestpar$cost=df.tune2$best.parameters$cost
    bestperf = df.tune2$best.performance
    pdf(paste("./plots/svm/poly2_", filename, "_tune.pdf", sep=""))
    plot(perf$cost, perf$error, type="l", ylab="Test error estimate", xlab="Cost", main=paste(main, " : SVM cost optimisation"))
    dev.off()
    write.csv(
        bestpar,
        file=paste("./csv/svm/", filename, "_poly2_svm_bestpar.csv", sep="")
    )
    write.csv(
        bestperf,
        file=paste("./csv/svm/", filename, "_poly2_svm_bestperf.csv", sep="")
    )
    
    # TO DO if we got time to waste
    # - coef0 optimisation
    
    return (bestpar)
}


assignclass = function(x) {
    # EXPERIMENTAL : TO BE USED IN AN APPLY( , , FUN )
    # testing to build a function that assigns the most 
    # predicted class to an individual
    return (sample(names(which(table(x) == max(table(x)))),1))
}

svm_conf_matrix = function(X, y, cost, gamma, kernel, filename="", main="") {
    # Use a 6-fold cross validation method to build a prediction
    # and the associated confusion matrix
    
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
        if (kernel == "sigmoid") {
            model = svm(
                as.factor(y)~., 
                data=df[-folds[[k]],], 
                kernel=kernel, 
                cost=cost,
                gamma=gamma
            )
        } else {
            model = svm(
                as.factor(y)~., 
                data=df[-folds[[k]],], 
                kernel=kernel, 
                cost=cost,
                gamma=gamma,
                degree=1
            )
        }
        preds = predict(model, newdata=df[folds[[k]],])
        
        # build the confusion matrix
        confs[,,k] = table(df[folds[[k]],]$y, preds)
        conf_matrix = conf_matrix + confs[,,k]
        
        # measure the test error
        tst_errors[k] = length(which(df[folds[[k]],]$y != preds))/length(df[folds[[k]],]$y)
    }
    
    # save the stats
    write.csv(conf_matrix, file=paste("./csv/svm/conf_matrix_", filename, kernel, ".csv", sep=""))
    write.csv(mean(tst_errors), file=paste("./csv/svm/tst_err_", filename, kernel,  ".csv", sep=""))

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
    
    return (conf_matrix)
}
