# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

library(MASS)
library(xlsx) # Easy xls export 


fa = function(X, y, filename="", main="") {
    l = list()
    l$lda = lda(X, y)
    l$scaling = l$lda$scaling
    l$x = X %*% l$scaling
    l$x.scaled = scale(l$x)
    
    # Plot over the first factorial plane
    pdf(paste("./plots/fa/fa_", filename, "_ffp.pdf", sep=""))
    plot(
        l$x.scaled[,1],
        l$x.scaled[,2],
        xlab="X1",
        ylab="X2",
        main=paste(main, ": FA result on the first factorial plane"),
        col=y
    )
    dev.off()
    return (l)
}
