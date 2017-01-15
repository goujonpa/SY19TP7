# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

library(neuralnet) # Neural networks
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
  #SIZE = c(c(10, 10), c(15, 15),
  #         c(10, 10, 10),
  #         c(10, 10, 10, 10))
  SIZE = c(c(10, 10))
  folds = createFolds(y, k=6)
  
  perf = data.frame(
    size=integer(),
    decay=double(),
    error=double(),
    sd=double(),
    stringsAsFactors = F
  )
  errors = vector(length=6)
  
  ## Instead of decay, this time we use the learning rate
  
  for(lr in seq(0, 1, 0.1)){
    for (size in SIZE) {
      for (i in 1:6) {
        
        ########
        # THERE IS A BUG WITH THE FORMULA CHECKER
        # http://stackoverflow.com/questions/17794575/error-in-terms-formulaformula-in-formula-and-no-data-argument
        ########
        
        ### We are thus obliged to fold manually
        folded = df[-folds[[i]],]
        
        #### WE STRINGIFY THE FORMULA 
        n <- names(df)
        folded = as.data.frame(df[-folds[[i]],])
        fact_y = as.factor(folded$y)
        f <- as.formula(paste("folded$y ~", paste(paste("folded$",n[!n %in% "y"]), collapse = " + ")))
        
        model = neuralnet(
          f,
          data=as.data.frame(folded),
          hidden=size,
          learningrate = lr,
          algorithm = "backprop",
          linear.output = F
        )
        
        ##### THERE IS A BUG WITH COMPUTE
        # http://stackoverflow.com/a/35051097/1895443
        pred = compute(model, as.matrix(df[folds[[i]],]))
        errors[i] = length(which(pred != df[folds[[i]],]$y))/length(df[folds[[i]],]$y)
      }
      perf[nrow(perf)+1,] = c(size, lr, mean(errors), sd(errors))
    }
    pdf(paste("./plots/neuraln/", filename, "_neuraln_net.pdf", sep=""))
    gwplot(model, "best")
    dev.off()
  }
  
  bestpar = perf[which(perf$error == min(perf$error)),]
  bestperf = bestpar$error
  
  # save performances
  write.csv(
    perf, 
    file=paste("./csv/neuraln/", filename, "_neuraln_perfomances.csv", sep="")
  )
  
  write.csv(
    bestpar,
    file=paste("./csv/neuraln/", filename, "_neuraln_bestpar.csv", sep="")
  )
  
  write.csv(
    bestperf,
    file=paste("./csv/neuraln/", filename, "_neuraln_bestperf.csv", sep="")
  )
  
  # plot the tuning perfs
  pdf(paste("./plots/neuraln/", filename, "_neuraln_tune1.pdf", sep=""))
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
      perf[which(perf$size ==pred[i]),]$error,
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

nn_conf_matrix = function(X, y, size, filename="", main="") {
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
  
  n <- names(df)
  fact_y = as.factor(y)
  f <- as.formula(paste("fact_y ~", paste(n[!n %in% "y"], collapse = " + ")))
  for (k in 1:6) {
    # fit and predict
    n <- names(df)
    folded = as.data.frame(df[-folds[[k]],])
    fact_y = as.factor(folded$y)
    f <- as.formula(paste("folded$y ~", paste(paste("folded$",n[!n %in% "y"]), collapse = " + ")))
    model = neuralnet(
      f, 
      data=folded,
      algorithm="backprop",
      err.fct = "ce"
    )
    preds = compute(model, newdata=df[folds[[k]],])
    
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
