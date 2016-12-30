# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Facial expression recognition

# custom principal component function
# X : data matrix

pcf = function(X, y) {
    # NOTE : from web sources, prcomp is prefered
    
    pc = prcomp(X)
    pdf("./plots/prcomp_xps.pdf")
    plot(pc)
    dev.off()
    pdf("./plots/prcomp_xps2.pdf")
    biplot(pc)
    dev.off()
    
    # proportions of variance explained
    pc$vars = pc$sdev^2
    pc$varsprop = pc$vars / sum(pc$vars)
    pc$cumvarsprop = cumsum(pc$varsprop)
    
    pdf("./plots/pca1.pdf")
    plot(pc$varsprop, 
         xlab = "Principal Component", 
         ylab="Proportion of variance explained",
         main="PCA - Proportion of variance explained",
         ylim = c(0,0.2), type="h")
    dev.off()
    
    pdf("./plots/pca2.pdf")
    plot(pc$varsprop[1:50], 
         xlab = "Principal Component", 
         ylab="Proportion of variance explained",
         main="PCA - Proportion of variance explained (PC1 to PC50)",
         ylim = c(0,0.2), type="h")
    dev.off()
    
    pdf("./plots/pca3.pdf")
    plot(pc$cumvarsprop, 
         xlab = "Principal Component", 
         ylab="Cumulative proportion of variance explained",
         main="PCA - cumulative proportion of variance explained",
         ylim = c(0,1), type="l")
    dev.off()
    
    # TO DO : IMPROVE CUMULATIVE VARIANCE PLOT
    # pc$cumvarsprop
    # text(100, pc$cumvarsprop[15], labels = as.character(pc$cumvarsprop[15]))
    # abline(v=15)
    # abline(h=pc$cumvarsprop[15])
    
    # >>>>> FIRST TWO PRINCIPAL COMPONENT PLOT
    pdf("./plots/2prcomp.pdf")
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
    
    # >>>>> PCA VIS
    #plot(prcf[,1], prcf[,2], col=prcf$y)
    
    
    return (pc)
}