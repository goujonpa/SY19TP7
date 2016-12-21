# Paul GOUJON & Jo COLINA
# UTC - SY19 - TP7

# Display functions

disp = function(m, r, c) {
    I = matrix(m, r, c)
    I1 = apply(I,1,rev) 
    image(t(I1), col=gray(0:255/255))
}