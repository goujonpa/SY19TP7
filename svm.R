# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial recognition

# SVM analysis function


library(e1071) # SVM
library(pROC)
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

svm_sigmoid_analysis(X, y, filename="", main="") {
    # function to be used when the first analysis tells us that 
    # the sigmoid svm model is the best one to use
    
    # >>>>> Initial DATA FRAME
    df = cbind(as.data.frame(X), y)
    
    # cost tuning
    df.tune = tune(
        svm, 
        as.factor(y)~., 
        data=df,
        ranges=list(
            cost=(c(50:250)/100)
        ), 
        kernel="sigmoid"
    )
    perf = df.tune$performances
    pdf(paste("./plots/svm/sigmoid_", filename, "tune.pdf", sep=""))
    plot(perf$cost, perf$error, type="l", ylab="Test error estimate", xlab="Cost", main=paste(main, " : SVM cost optimisation"))
    dev.off()
    
    write.csv(
        df.tune$best.parameters,
        file=paste("./csv/svm/", filename, "_sigmoidsvm_bestpar.csv", sep="")
    )
    
    write.csv(
        df.tune$best.performance,
        file=paste("./csv/svm/", filename, "_sigmoidsvm_bestperf.csv", sep="")
    )
    
    # BONUS : TO DO if we have time to waste :
    # - Gamma optimisation
    # - coef0 optimisation
    
    new_cost = df.tune$best.parameters$cost
        
    return (new_cost)
}

svm_conf_matrix = function(X, y, filename="", main="") {
    # Use a 6-fold cross validation method to build a prediction
    # and the associated confusion matrix
    
    # >>>>> Initial DATA FRAME
    
    
    
}
    
    
    
    
    
