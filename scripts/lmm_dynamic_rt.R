# Trial by trial prediction using Linear Mixed Model
# Predictors are dynamic MAG only
# predict response time

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

##################################### rt #######################################
# random intercept
MLexamp <- lmer(rt ~ avgDepthSol
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
plot(pred, test$rt,
     xlab='Pred log(rt)', ylab='True log(rt)',
     main='Pred vs True Log(rt) LMM Random Intercept', 
     cex=0.3)
lines(test$rt, test$rt, 
      lwd=2)
mtext(text=paste(c("R2 =", r.squaredGLMM(MLexamp)[2]), collapse = " "), 
      side=3)
# line plots
plot(test$rt, 
     main="LMM Random Intercepts", 
     xlab="Trials", ylab="Log(rt)",
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
MLexamp11 <- lmer(rt ~ avgDepthSol 
           + avgEdgeSol 
           + avgMaxCycleSol 
           + avgcNodeSol 
           + avgnCycleSol 
           + backMoveSol 
           + edgeRate 
           + nodeRate 
           + (1 | sublist),
           data = train)
display(MLexamp11)
summary(MLexamp11)
r.squaredGLMM(MLexamp11)
pred11 = predict(MLexamp11, test)
# scatter plot
plot(pred11, test$rt,
     xlab='Pred log(rt)', ylab='True log(rt)',
     main='Model Found Log(rt) LMM Random Intercept', 
     cex=0.3)
lines(test$rt, test$rt, 
      lwd=2)
mtext(text=paste(c("R2 =", r.squaredGLMM(MLexamp11)[2]), collapse = " "), 
      side=3)




# random slope and intercept independent
MLexamp2 <- lmer(rt ~ avgDepthSol + (0 + avgDepthSol | sublist) 
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
plot(pred2, test$rt,
     xlab='Pred log(rt)', ylab='True log(rt)',
     main='Pred vs True Log(rt) LMM Rand Slope & Intercept indep', 
     cex=0.3)
lines(test$rt, test$rt, 
      lwd=2)
mtext(text=paste(c("R2 =", r.squaredGLMM(MLexamp2)[2]), collapse = " "), 
      side=3)
# line plots
plot(test$rt, 
     main="LMM Random Slopes & Intercepts indep", 
     xlab="Trials", ylab="Log(rt)",
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
MLexamp22 <- lmer(rt ~ avgDepthSol 
                  + avgEdgeSol 
                  + avgMaxCycleSol 
                  + avgcNodeSol 
                  + avgnCycleSol 
                  + backMoveSol 
                  + edgeRate 
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
pred11 = predict(MLexamp22, test)
# scatter plot
plot(pred11, test$rt,
     xlab='Pred log(rt)', ylab='True log(rt)',
     main='Model Found Log(rt) LMM Random Intercept', 
     cex=0.3)
lines(test$rt, test$rt, 
      lwd=2)
mtext(text=paste(c("R2 =", r.squaredGLMM(MLexamp22)[2]), collapse = " "), 
      side=3)





# random slope and intercept correlated
MLexamp3 <- lmer(rt ~ avgDepthSol + (1 + avgDepthSol | sublist) 
                 + avgEdgeSol + (1 + avgEdgeSol | sublist)
                 + avgMaxCycleSol + (1 + avgMaxCycleSol | sublist)
                 + avgNodeSol + (1 + avgNodeSol | sublist)
                 + avgcNodeSol + (1 + avgcNodeSol | sublist)
                 + avgnCycleSol + (1 + avgnCycleSol | sublist)
                 + backMoveSol + (1 + backMoveSol | sublist)
                 + edgeRate + (1 + edgeRate | sublist)
                 + nodeRate + (1 + nodeRate | sublist)
                 + unsafeSol + (1 + unsafeSol | sublist),
                 data = train,
                 control = lmerControl(optimizer ='optimx', 
                                       optCtrl=list(method='L-BFGS-B')))
display(MLexamp3)
summary(MLexamp3)
r.squaredGLMM(MLexamp3)
pred3 = predict(MLexamp3, test)
# scatter plot
plot(pred3, test$rt,
     xlab='Pred log(rt)', ylab='True log(rt)',
     main='Pred vs True Log(rt) LMM Rand Slope & Intercept corr', 
     cex=0.3)
lines(test$rt, test$rt, 
      lwd=2)
mtext(text=paste(c("R2 =", r.squaredGLMM(MLexamp3)[2]), collapse = " "), 
      side=3)
# line plots
plot(pred3, 
     main="LMM Random Slopes & Intercepts corr", 
     xlab="Trials", ylab="Log(rt)",
     type='l', col='blue', xlim=c(1400,1600), lwd=0.5)
lines(test$rt, type='l', col='red', xlim=c(1400,1600), lwd=0.5)
legend(1550, 14, legend=c("Target", "Predict"),
       col=c("red", "blue"), lty=1, cex=0.8,
       box.lty=0)
mtext(text=paste(c("R2 =", r.squaredGLMM(MLexamp3)[2]), collapse = " "), 
      side=3)
# stepwise regression 
step_result3 <- step(MLexamp3)
step_result3


# random slope and intercept independent, single regressor
MLregr1 <- lmer(rt ~ avgDepthSol + (0 + avgDepthSol | sublist) 
                 + (1 | sublist),
                 data = train,
                 REML=F)
MLregr2 <- lmer(rt ~ avgEdgeSol + (0 + avgEdgeSol | sublist)
                + (1 | sublist),
                data = train,
                REML=F)
MLregr3 <- lmer(rt ~ avgMaxCycleSol + (0 + avgMaxCycleSol | sublist)
                + (1 | sublist),
                data = train,
                REML=F)
MLregr4 <- lmer(rt ~ avgNodeSol + (0 + avgNodeSol | sublist)
                + (1 | sublist),
                data = train,
                REML=F)
MLregr5 <- lmer(rt ~ avgcNodeSol + (0 + avgcNodeSol | sublist)
                + (1 | sublist),
                data = train,
                REML=F)
MLregr6 <- lmer(rt ~ avgnCycleSol + (0 + avgnCycleSol | sublist)
                + (1 | sublist),
                data = train,
                REML=F)
MLregr7 <- lmer(rt ~ backMoveSol + (0 + backMoveSol | sublist)
                + (1 | sublist),
                data = train,
                REML=F)
MLregr8 <- lmer(rt ~ edgeRate + (0 + edgeRate | sublist)
                + (1 | sublist),
                data = train,
                REML=F)
MLregr9 <- lmer(rt ~ nodeRate + (0 + nodeRate | sublist)
                + (1 | sublist),
                data = train,
                REML=F)
MLregr10 <- lmer(rt ~ unsafeSol + (0 + unsafeSol | sublist)
                + (1 | sublist),
                data = train,
                REML=F)
r.squaredGLMM(MLregr1)
r.squaredGLMM(MLregr2)
r.squaredGLMM(MLregr3)
r.squaredGLMM(MLregr4)
r.squaredGLMM(MLregr5)
r.squaredGLMM(MLregr6)
r.squaredGLMM(MLregr7)
r.squaredGLMM(MLregr8)
r.squaredGLMM(MLregr9)
r.squaredGLMM(MLregr10)
# plot the most important regressor
display(MLregr9)
summary(MLregr9)
r.squaredGLMM(MLregr9)
pred = predict(MLregr9, test)
# scatter plot
plot(pred, test$rt,
     xlab='Pred log(rt)', ylab='True log(rt)',
     main='Pred vs True Log(rt) LMM Rand Slope & Int indep', 
     cex=0.3)
lines(test$rt, test$rt, 
      lwd=2)
mtext(text=paste(c("nodeRate only, R2 =", r.squaredGLMM(MLregr9)[2]), collapse = " "), 
      side=3)
# line plot
plot(test$rt, 
     main="LMM Random Slopes & Intercepts indep regressor9", 
     xlab="Trials", ylab="Log(rt)",
     type='l', col='red', xlim=c(1400,1600), lwd=0.5)
lines(pred, type='l', col='blue', xlim=c(1400,1600), lwd=0.5)
legend(1400, 5.9, legend=c("Target", "Predict"),
       col=c("red", "blue"), lty=1, cex=0.8,
       box.lty=0)
mtext(text=paste(c("R2 =", r.squaredGLMM(MLregr9)[2]), collapse = " "), 
      side=3)
