# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

library(MASS)

fa = function(X, y, filename="", main="") {
    l = list()
    l$lda = lda(X, y)
    l$x = X %*% l$lda$scaling
    
    # Plot over the first factorial plane
    pdf(paste("./plots/fa/fa_", filename, "_ffp.pdf", sep=""))
    plot(
        l$x[,1],
        l$x[,2],
        xlab="X1",
        ylab="X2",
        main=paste(main, ": FA result on the first factorial plane"),
        col=y
    )
    dev.off()
    return (l)
}
