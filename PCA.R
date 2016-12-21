library(MASS)
load("C:\\Users\\Jo Colina\\Documents\\UTC\\SY19\\TP7\\DATA\\data_expressions.rdata")

## 
# Normalizes color of image
# To make it black & white
# To check if 0 spots are
# on the exact same place
##
normalizecolor = function(x){
  for(i in 1:length(x)){
    if(x[i] != 0){
      x[i] = 1
    }
  } 
  return (x)
}

##
# Displays the image for the given index
##
displayImage = function(index, normal){
  I<-matrix(X[index,],60,70)
  I1 <- apply(I, 1, rev)
  if(normal){
    image(t(I1),col=gray(0:255 / 255))  
  }
  
  
  I1 <- apply(I1, 1, normalizecolor)
  image(I1, col=c(0, 1))
}


Xnorm = X[,which(X[1,] != 0)]

## VERIFICATION Xnorm[1, which(Xnorm[1,] == 0)] 
ppca = prcomp(Xnorm, scale = TRUE)
spca = summary(ppca)


## prcomp PCA
# Plot to find best number of variables
# depending on variance explanation
vars <- apply(ppca$x, 2, var)  
props <- vars / sum(vars)
cumprop = cumsum(props)
plot(cumprop)
abline(v = 25, h=cumprop[25])
legend(x=100, y =cumprop[25], legend=paste(as.character(round(cumprop[25], 4)), "%", sep=" "))
abline(v = 50, h=cumprop[50])
legend(x=100, y =cumprop[50], legend=paste(as.character(round(cumprop[50], 4)), "%", sep=" "))

xprin = ppca$x[, 1:25]
df = as.data.frame(cbind(xprin, y))
colnames(df)[26] = "y"

## LDA
model = lda(y~., data=df)
lda.pred = predict(model, newdata=df)


