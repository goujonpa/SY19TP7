# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition
library(xlsx) # Easy xls export 


pca = function(X, y) {
    # NOTE : from web sources, prcomp is prefered
    
    pc = prcomp(X)
    
    # explained var plot
    pdf("./plots/pc/pc_plot.pdf")
    plot(pc)
    dev.off()
    
    # ugly pc plot
    pdf("./plots/pc/pc_biplot.pdf")
    biplot(pc)
    dev.off()

    # proportions of variance explained 
    # custom calculation
    pc$vars = pc$sdev^2
    pc$varsprop = pc$vars / sum(pc$vars)
    pc$cumvarsprop = cumsum(pc$varsprop)

    # proportion of variance explained, every PC
    pdf("./plots/pc/pc_propvarexp.pdf")
    plot(pc$varsprop,
         xlab = "Principal Component",
         ylab="Proportion of variance explained",
         main="PCA - Proportion of variance explained",
         ylim = c(0,0.2), type="h")
    dev.off()

    # proportion of variance explained, 50 first PCs
    pdf("./plots/pc/pc_propvarexp2.pdf")
    plot(pc$varsprop[1:50],
         xlab = "Principal Component",
         ylab="Proportion of variance explained",
         main="PCA - Proportion of variance explained (PC1 to PC50)",
         ylim = c(0,0.2), type="h")
    dev.off()

    # cumulation proportion of variance explained
    pdf("./plots/pc/pc_cumpropvar.pdf")
    plot(pc$cumvarsprop,
        xlab = "Principal Component",
        ylab="Cumulative proportion of variance explained",
        main="PCA - cumulative proportion of variance explained",
        ylim = c(0,1), type="l")
        text(100, pc$cumvarsprop[15] - 0.03, 
             labels=as.character(paste("15 - ", round(pc$cumvarsprop[15], 4), '%')))
        abline(v=15)
        abline(h=pc$cumvarsprop[15])
        text(
            100, pc$cumvarsprop[25] - 0.03, 
            labels=as.character(paste("25 - ", round(pc$cumvarsprop[25], 4), '%')))
        abline(v=25)
        abline(h=pc$cumvarsprop[25])
        text(
            100, pc$cumvarsprop[50] - 0.03, 
            labels=as.character(paste("50 - ", round(pc$cumvarsprop[50], 4), '%')))
        abline(v=50)
        abline(h=pc$cumvarsprop[50])
        dev.off()

    # first two principal component plot
    pdf("./plots/pc/pc_2prcomp.pdf")
    plot(
        pc$x[,1],
        pc$x[,2],
        col=as.numeric(y),
        main="Two Principal Component",
        xlab="First principal component",
        ylab="Second principal component"
    )
    dev.off()

    # TEST : SHOWING VARIANCE, TO REUSE OR DELETE
    # sdv = apply(Xpsf[which(Xpsf$y==as.character(j)),-3661], 2, sd)
    # for (i in 1:3660) {
    #     segments(i, ys[i]+sdv[i], i, ys[i]-sdv[i])
    # }

    return (pc)
}