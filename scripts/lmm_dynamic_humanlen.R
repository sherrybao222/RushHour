# Trial by trial prediction using Linear Mixed Model
# Predictors are dynamic MAG only
# predict humane solution length

# install library
install.packages("lme4")
install.packages("arm")
install.packages("sjmisc")
install.packages("MuMIn")
install.packages("lmerTest")
install.packages("optimx")

# initialize
rm(list = ls())
library(lme4)  # load library
library(arm)  # convenience functions for regression in R
library(sjmisc)
library(MuMIn)
library(lmerTest)
library(optimx)
train <- read.csv(file="/Users/chloe/Desktop/train.csv")
test <- read.csv(file="/Users/chloe/Desktop/test.csv")
summary(train)
summary(test)
head(train)
head(test)

############################## static MAG and optlen ###########################

# humanlen random intercept
# fixed-effect model matrix is rank deficient so dropping 1 column / coefficient
MLexamp <- lmer(humanlen ~ c_incycle
                + countcycle
                + countscc
                + depth
                + e2n
                + edges
                + en
                + enp
                + gcluster
                + lcluster
                + maxcycle
                + maxscc
                + ndepth
                + ncc
                + nodes
                + pnc
                + avgDepthSol
                + avgEdgeSol 
                + avgMaxCycleSol
                + avgNodeSol 
                + avgcNodeSol 
                + avgnCycleSol 
                + backMoveSol 
                + edgeRate 
                + nodeRate 
                + unsafeSol
                + (1 | sublist),
                data = train)
display(MLexamp)
summary(MLexamp)
r.squaredGLMM(MLexamp)
pred = predict(MLexamp, test)
# scatter plot
plot(pred, test$humanlen,
     xlab='Pred log(humanlen)', ylab='True log(humanlen)',
     main='All MAG LMM Random Intercept', 
     cex=0.3)
lines(test$humanlen, test$humanlen, 
      lwd=2)
mtext(text=paste(c("R2 =", r.squaredGLMM(MLexamp)[2]), collapse = " "), 
      side=3)
# line plots
plot(test$humanlen, 
     main="All MAG LMM Random Intercepts", 
     xlab="Trials", ylab="Log(humanlen)",
     type='l', col='red', xlim=c(1400,1600),lwd=0.5)
lines(pred, type='l', col='blue', xlim=c(1400,1600), lwd=0.5)
legend(1550, 14, legend=c("Target", "Predict"),
       col=c("red", "blue"), lty=1, cex=0.8,
       box.lty=0)
mtext(text=paste(c("R2 =", r.squaredGLMM(MLexamp)[2]), collapse = " "), 
      side=3)
# stepwise regression 
step_result <- step(MLexamp)
step_result
# model found
# fixed-effect model matrix is rank deficient so dropping 1 column / coefficient
MLexamp11 <- lmer(humanlen ~ c_incycle 
                  + countcycle 
                  + countscc 
                  + depth 
                  + e2n 
                  + edges 
                  + maxcycle 
                  + maxscc 
                  + ncc 
                  + pnc 
                  + avgDepthSol 
                  + avgEdgeSol 
                  + nodeRate 
                  + (1 | sublist),
                  data = train)
                    
display(MLexamp11)
summary(MLexamp11)
r.squaredGLMM(MLexamp11)
pred11 = predict(MLexamp11, test)
# scatter plot
plot(pred11, test$humanlen,
     xlab='Pred log(humanlen)', ylab='True log(humanlen)',
     main='Model Found All MAG LMM Random Intercept', 
     cex=0.3)
lines(test$humanlen, test$humanlen, 
      lwd=2)
mtext(text=paste(c("R2 =", r.squaredGLMM(MLexamp11)[2]), collapse = " "), 
      side=3)



# random slope and intercept independent
MLexamp2 <- lmer(humanlen ~ c_incycle #+ (0 + c_incycle | sublist)
                  + countcycle #+ (0 + countcycle | sublist)
                  + countscc #+ (0 + countscc | sublist)
                  + depth #+ (0 + depth | sublist)
                  + e2n #+ (0 + e2n | sublist)
                  + edges #+ (0 + edges | sublist)
                  + en #+ (0 + en | sublist)
                  + enp #+ (0 + enp | sublist)
                  + gcluster #+ (0 + gcluster | sublist)
                  + lcluster #+ (0 + lcluster | sublist)
                  + maxcycle #+ (0 + maxcycle | sublist)
                  + maxscc #+ (0 + maxscc | sublist)
                  + ndepth #+ (0 + ndepth | sublist)
                  + ncc #+ (0 + ncc | sublist)
                  + nodes #+ (0 + nodes | sublist)
                  + pnc #+ (0 + pnc | sublist)
                  + optlen
                  + avgDepthSol + (0 + avgDepthSol | sublist) 
                  + avgEdgeSol + (0 + avgEdgeSol | sublist)
                  + avgMaxCycleSol + (0 + avgMaxCycleSol | sublist)
                  + avgNodeSol + (0 + avgNodeSol | sublist)
                  + avgcNodeSol + (0 + avgcNodeSol | sublist)
                  + avgnCycleSol + (0 + avgnCycleSol | sublist)
                  + backMoveSol + (0 + backMoveSol | sublist)
                  + edgeRate + (0 + edgeRate | sublist)
                  + nodeRate + (0 + nodeRate | sublist)
                  + unsafeSol + (0 + unsafeSol | sublist)
                  + (1 | sublist),
                  data = train,
                  REML=F)
display(MLexamp2)
summary(MLexamp2)
r.squaredGLMM(MLexamp2)
pred2 = predict(MLexamp2, test)
# scatter plot
plot(pred2, test$humanlen,
     xlab='Pred log(humanlen)', ylab='True log(humanlen)',
     main='All MAG LMM Rand Slope & Int indep', 
     cex=0.3)
lines(test$humanlen, test$humanlen, 
      lwd=2)
mtext(text=paste(c("R2 =", r.squaredGLMM(MLexamp2)[2]), collapse = " "), 
      side=3)
# line plots
plot(test$humanlen, 
     main="All MAG LMM Rand Slopes & Int indep", 
     xlab="Trials", ylab="Log(humanlen)",
     type='l', col='red', xlim=c(1400,1600), lwd=0.5)
lines(pred2, type='l', col='blue', xlim=c(1400,1600), lwd=0.5)
legend(1550, 14, legend=c("Target", "Predict"),
       col=c("red", "blue"), lty=1, cex=0.8,
       box.lty=0)
mtext(text=paste(c("R2 =", r.squaredGLMM(MLexamp2)[2]), collapse = " "), 
      side=3)
# stepwise regression
step_result2 <- step(MLexamp2)
step_result2
# found model
MLexamp22 <- lmer(humanlen ~ c_incycle 
                  + countcycle 
                  + countscc 
                  + depth 
                  + e2n 
                  + edges 
                  + maxcycle 
                  + maxscc 
                  + ncc 
                  + pnc 
                  + avgDepthSol 
                  + avgEdgeSol 
                  + nodeRate 
                  + (0 + avgEdgeSol | sublist) 
                  + (0 + avgnCycleSol | sublist) 
                  + (0 + nodeRate | sublist) 
                  + (1 | sublist),
                  data = train,
                  REML=F)
display(MLexamp22)
summary(MLexamp22)
r.squaredGLMM(MLexamp22)
pred22 = predict(MLexamp22, test)
# scatter plot
plot(pred22, test$humanlen,
     xlab='Pred log(humanlen)', ylab='True log(humanlen)',
     main='Model Found All MAG LMM Random Intercept', 
     cex=0.3)
lines(test$humanlen, test$humanlen, 
      lwd=2)
mtext(text=paste(c("R2 =", r.squaredGLMM(MLexamp22)[2]), collapse = " "), 
      side=3)

