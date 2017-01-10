# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

source("./display.R") # custom display func

exp_mean = function(X, y, EXPRESSIONS) {
    l = list()
    for (i in 1:6) {
        pdf(paste("./plots/meanface_", as.character(i), ".pdf", sep=""))
        Z = apply(
            X[which(y == as.character(i)),],
            2,
            mean
        )
        disp(Z, 60, 70, paste("Image moyenne : ", EXPRESSIONS[i], sep=""))
        dev.off()
        l[[i]] = Z
    }
    return (l)
}
