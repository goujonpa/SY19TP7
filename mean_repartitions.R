# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

mean_rep = function (X, y, EXPRESSIONS){
    l = list()
    for (j in 1:6) {
        pdf(paste("./plots/mean_repartition_", as.character(j), ".pdf", sep=""))
        Z = apply(
            X[which(y == as.character(j)),],
            2,
            mean
        )
        plot(
            1:ncol(X),
            Z,
            col=j,
            main=paste("Répartition moyenne : ", EXPRESSIONS[j], sep=""),
            xlab="Component",
            ylab="Mean Value"
        )
        dev.off()
        l[[j]] = Z
    }
    return (l)
}