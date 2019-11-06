fun <-function(x) (exp(2*sigma*x)-1) / (exp(2*sigma*x)+1)

#sigma.seq <- c(0.2, 0.4, 0.5, 1,2,5)

sigma.seq <- c(seq(0.2, 1.8, 0.2))
cols <- viridis::viridis(length(sigma.seq))

sigma <- sigma.seq[1]
curve(fun, xlim = c(-5,5), ylim = c(-1.5, 1.5)  ,col=cols[1], n=1000)
grid()

for(i in 2:length(sigma.seq)){
  sigma <- sigma.seq[i]
  curve(fun, xlim = c(-5,5), col=cols[i], add = T, n=1000)
}

legend("topleft", legend = sigma.seq, lwd=rep(1,length(sigma.seq)), col=cols, cex = .5)