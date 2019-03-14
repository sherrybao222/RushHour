# Trial by trial prediction using Linear Mixed Model
# Predictors are dynamic MAG and static MAG, and optlen as well
# predict humane solution length

# install library
install.packages("lme4")
install.packages("arm")
install.packages("sjmisc")
install.packages("MuMIn")
install.packages("lmehumanlenest")
install.packages("optimx")
install.packages("viridis")

# initialize
rm(list = ls())
library(lme4)  # load library
library(nlme)
library(arm)  # convenience functions for regression in R
library(sjmisc)
library(MuMIn)
library(lmertest)
library(optimx)
library(ggplot2)
library(viridis)
train <- read.csv(file="/Users/chloe/Documents/RushHour/state_model/in_data_trials/train.csv")
test <- read.csv(file="/Users/chloe/Documents/RushHour/state_model/in_data_trials/test.csv")
summary(train)
summary(test)
head(train)
head(test)

# preprocess data
train$humanlen <- log(train$humanlen)
test$humanlen <- log(test$humanlen)

############################## static MAG and optlen ###########################
# humanlen random intercept
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
                + optlen
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
#R2m       R2c
#[1,] 0.4925919 0.5893697
pred = predict(MLexamp, test)
# scatter plot
# plot(pred, test$humanlen,
#      xlab='Pred log(humanlen)', ylab='True log(humanlen)',
#      main='All MAG LMM Random Intercept', 
#      cex=0.3)
# lines(test$humanlen, test$humanlen, 
#       lwd=2)
# mtext(text=paste(c("R2 =", r.squaredGLMM(MLexamp)[2]), collapse = " "), 
#       side=3)
# line plots
# plot(test$humanlen, 
#      main="All MAG LMM Random Intercepts", 
#      xlab="Trials", ylab="Log(humanlen)",
#      type='l', col='red', xlim=c(1400,1600),lwd=0.5)
# lines(pred, type='l', col='blue', xlim=c(1400,1600), lwd=0.5)
# legend(1550, 14, legend=c("Target", "Predict"),
#        col=c("red", "blue"), lty=1, cex=0.8,
#        box.lty=0)
# mtext(text=paste(c("R2 =", r.squaredGLMM(MLexamp)[2]), collapse = " "), 
#       side=3)
# stepwise regression 
step_result <- step(MLexamp)
step_result
# model found
MLexamp11 <- lmer(humanlen ~ c_incycle 
                  + countcycle 
                  + countscc 
                  + depth 
                  + e2n 
                  + edges 
                  + en 
                  + gcluster 
                  + lcluster 
                  + maxcycle 
                  + maxscc 
                  + ncc 
                  + nodes 
                  + pnc 
                  + optlen 
                  + avgDepthSol 
                  + avgEdgeSol 
                  + avgMaxCycleSol 
                  + nodeRate 
                  + (1 | sublist),
                  data = train,
                  REML=F)
display(MLexamp11)
summary(MLexamp11)
r.squaredGLMM(MLexamp11)
#R2m       R2c
#[1,] 0.4951608 0.5900379
pred11 = predict(MLexamp11, test)
# scatter plot
# plot(pred11, test$humanlen,
#      xlab='Pred log(humanlen)', ylab='True log(humanlen)',
#      main='Model Found All MAG LMM Random Intercept', 
#      cex=0.3)
# lines(test$humanlen, test$humanlen, 
#       lwd=2)
# mtext(text=paste(c("R2 =", r.squaredGLMM(MLexamp11)[2]), collapse = " "), 
#       side=3)


# random intercept and random slopes independent
MLexamp2 <- lmer(humanlen ~ c_incycle + (0 + c_incycle | sublist)
                  + countcycle + (0 + countcycle | sublist)
                  + countscc + (0 + countscc | sublist)
                  + depth + (0 + depth | sublist)
                  + e2n + (0 + e2n | sublist)
                  + edges + (0 + edges | sublist)
                  + en + (0 + en | sublist)
                  + gcluster + (0 + gcluster | sublist)
                  + lcluster + (0 + lcluster | sublist)
                  + maxcycle + (0 + maxcycle | sublist)
                  + maxscc + (0 + maxscc | sublist)
                  + ncc + (0 + ncc | sublist)
                  + nodes + (0 + nodes | sublist)
                  + pnc + (0 + pnc | sublist)
                  + optlen + (0 + optlen | sublist)
                  + avgDepthSol + (0 + avgDepthSol | sublist)
                  + avgEdgeSol + (0 + avgEdgeSol | sublist)
                  + avgMaxCycleSol + (0 + avgMaxCycleSol | sublist)
                  + nodeRate + (0 + nodeRate | sublist)
                  + (1 | sublist),
                  data = train,
                  REML=F)
display(MLexamp2)
summary(MLexamp2)
r.squaredGLMM(MLexamp2)
# R2m       R2c
# [1,] 0.5052455 0.6036803
pred2 = predict(MLexamp2, test)
# stepwise regression 
step_result <- step(MLexamp2)
step_result
MLexamp22 <- lmer(humanlen ~ c_incycle 
                  + countcycle 
                  + countscc 
                  + depth 
                  + e2n 
                  + edges 
                  + en 
                  + gcluster + (0 + gcluster | sublist)
                  + lcluster 
                  + maxcycle 
                  + maxscc 
                  + ncc 
                  + nodes + (0 + nodes | sublist)
                  + pnc 
                  + optlen + (0 + optlen | sublist)
                  + avgDepthSol + (0 + avgDepthSol | sublist)
                  + avgEdgeSol + (0 + avgEdgeSol | sublist)
                  + avgMaxCycleSol + (0 + avgMaxCycleSol | sublist)
                  + nodeRate + (0 + nodeRate | sublist),
                  data = train,
                  REML=F)
display(MLexamp22)
summary(MLexamp22)
r.squaredGLMM(MLexamp22)
# R2m       R2c
# [1,] 0.5052556 0.6037029
pred22 = predict(MLexamp22, test)




################################### PLOT BEST MODE #################################
# prepare data
best_model <- MLexamp2
plot_dir1 <- "/Users/chloe/Documents/RushHour/state_model/out_model/slope-humanlen-ellipse-CI068-puz.png"
plot_dir2 <- "/Users/chloe/Documents/RushHour/state_model/out_model/slope-humanlen-ellipse-CI068-optlen.png"
plot_dir3 <- "/Users/chloe/Documents/RushHour/state_model/out_model/slope-humanlen-scatter-puz.png"
plot_dir4 <- "/Users/chloe/Documents/RushHour/state_model/out_model/slope-humanlen-scatter-optlen.png"
best_title <- "All MAG + optlen -> humanlen, LMM random int & slope"
pred_frame <- data.frame(pred2)
pred_frame
test_frame <- data.frame(test)
test_frame
ggdat <- cbind(test_frame, pred_frame)
ggdat$optlen <- as.character(as.numeric(ggdat$optlen))
class(ggdat$optlen)
ggdat$optlen
# ggplot ellipse by puzzle
p <- ggplot(ggdat, aes(pred, humanlen, colour=puzzle)) + 
  geom_point(size=1, alpha=0.6, show.legend=FALSE) + 
  labs(x = "Pred log(humanlen)") + 
  labs(y = "True log(humanlen)") + 
  labs(title = best_title, subtitle = paste(c("R2 =", r.squaredGLMM(best_model)[2]), collapse = " ")) + 
  geom_line(aes(pred, pred), alpha=0.5,show.legend = FALSE) +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5)) + 
  labs(caption = "coloured by puzzle") + 
  theme(panel.grid.minor=element_line(colour="grey", linetype=3, size=0.5)) +
  scale_y_continuous(minor_breaks = c(log(7),log(11),log(14),log(16)), breaks = seq(0,6,0.5)) +
  scale_x_continuous(minor_breaks = c(log(7),log(11),log(14),log(16)), breaks = seq(0,5,0.5)) +
  stat_ellipse(level=0.68, alpha=0.4, type = "norm", show.legend = FALSE)
p
ggsave(plot_dir1)
# ggplot ellipse group by puzzle, color by optlen
p <- ggplot(ggdat, aes(pred, humanlen, group=puzzle, colour=optlen)) + 
  geom_point(size=1, alpha=0.6, aes(group=puzzle, colour=optlen)) + 
  labs(x = "Pred log(humanlen)") + 
  labs(y = "True log(humanlen)") + 
  labs(title = best_title, subtitle = paste(c("R2 =", r.squaredGLMM(best_model)[2]), collapse = " ")) + 
  geom_line(aes(pred, pred), alpha=0.5,show.legend = FALSE) +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5)) + 
  labs(caption = "grouped by puzzle, coloured by optlen") + 
  theme(panel.grid.minor=element_line(colour="grey", linetype=3, size=0.5)) +
  scale_y_continuous(minor_breaks = c(log(7),log(11),log(14),log(16)), breaks = seq(0,6,0.5)) +
  scale_x_continuous(minor_breaks = c(log(7),log(11),log(14),log(16)), breaks = seq(0,5,0.5)) +
  stat_ellipse(level=0.68, alpha=0.4, type = "norm") + 
  scale_colour_brewer(palette = "Set2", breaks = c("16","14","11","7"))
p
ggsave(plot_dir2)
# ggplot scatter by puzzle
p <- ggplot(ggdat, aes(pred, humanlen, colour=puzzle)) + 
  geom_point(size=1, alpha=0.95, show.legend=FALSE)+ labs(x = "Pred log(humanlen)") + 
  labs(y = "True log(humanlen)") +
  labs(title = best_title, subtitle = paste(c("R2 =", r.squaredGLMM(best_model)[2]), collapse = " ")) + 
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5)) + 
  labs(caption = "coloured by puzzle") +
  geom_line(aes(pred, pred), show.legend = FALSE) +
  theme(panel.grid.minor=element_line(colour="grey", linetype=3, size=0.5)) +
  scale_y_continuous(minor_breaks = c(log(7),log(11),log(14),log(16)), breaks = seq(0,6,0.5)) +
  scale_x_continuous(minor_breaks = c(log(7),log(11),log(14),log(16)), breaks = seq(0,5,0.5))
p
ggsave(plot_dir3)
# ggplot scatter by optlen
p <- ggplot(ggdat, aes(pred, humanlen)) +
  geom_point(size=1, alpha=0.95, aes(group=optlen, colour=optlen)) + 
  scale_colour_brewer(palette = "Set2", breaks = c("16","14","11","7")) + 
  labs(x = "Pred log(humanlen)") + 
  labs(y = "True log(humanlen)") + 
  labs(title = best_title, subtitle = paste(c("R2 =", r.squaredGLMM(best_model)[2]), collapse = " ")) +
  labs(caption = "coloured by optlen") +
  theme(plot.title=element_text(hjust=0.5), plot.subtitle=element_text(hjust = 0.5)) +
  theme(panel.grid.minor=element_line(colour="grey", linetype=3, size=0.5)) +
  scale_y_continuous(minor_breaks = c(log(7),log(11),log(14),log(16)), breaks = seq(0,6,0.5)) +
  scale_x_continuous(minor_breaks = c(log(7),log(11),log(14),log(16)), breaks = seq(0,5,0.5)) + 
  geom_line(aes(pred, pred, colour=optlen), show.legend = FALSE, size = 1)
p
ggsave(plot_dir4)




























#################################### DISCARDED #####################################
# random slope and intercept independent
MLexamp2 <- lmer(humanlen ~ c_incycle + (0 + c_incycle | sublist)
                 + countcycle + (0 + countcycle | sublist)
                 + countscc + (0 + countscc | sublist)
                 + depth + (0 + depth | sublist)
                 + e2n + (0 + e2n | sublist)
                 + edges + (0 + edges | sublist)
                 + en + (0 + en | sublist)
                 + enp #+ (0 + enp | sublist)
                 + gcluster + (0 + gcluster | sublist)
                 + lcluster + (0 + lcluster | sublist)
                 + maxcycle + (0 + maxcycle | sublist)
                 + maxscc + (0 + maxscc | sublist)
                 + ndepth #+ (0 + ndepth | sublist)
                 + ncc + (0 + ncc | sublist)
                 + nodes + (0 + nodes | sublist)
                 + pnc + (0 + pnc | sublist)
                 + optlen + (0 + optlen | sublist)
                 + avgDepthSol + (0 + avgDepthSol | sublist) 
                 + avgEdgeSol + (0 + avgEdgeSol | sublist)
                 + avgMaxCycleSol + (0 + avgMaxCycleSol | sublist)
                 + avgNodeSol #+ (0 + avgNodeSol | sublist)
                 + avgcNodeSol #+ (0 + avgcNodeSol | sublist)
                 + avgnCycleSol #+ (0 + avgnCycleSol | sublist)
                 + backMoveSol #+ (0 + backMoveSol | sublist)
                 + edgeRate #+ (0 + edgeRate | sublist)
                 + nodeRate + (0 + nodeRate | sublist)
                 + unsafeSol #+ (0 + unsafeSol | sublist)
                 + (1 | sublist),
                 data = train,
                 REML=F)
display(MLexamp2)
summary(MLexamp2)
r.squaredGLMM(MLexamp2)
#R2m       R2c
#[1,] 0.5001883 0.6109378
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
                  + en 
                  + gcluster 
                  + lcluster 
                  + maxcycle 
                  + maxscc 
                  + ncc 
                  + nodes 
                  + pnc 
                  + optlen 
                  + avgDepthSol 
                  + avgEdgeSol 
                  + avgMaxCycleSol 
                  + nodeRate 
                  + (0 + avgDepthSol | sublist) 
                  + (0 + avgEdgeSol | sublist) 
                  + (0 + avgMaxCycleSol | sublist) 
                  + (0 + avgNodeSol | sublist) 
                  + (0 + avgcNodeSol | sublist) 
                  + (0 + avgnCycleSol | sublist) 
                  + (0 + backMoveSol | sublist) 
                  + (0 + nodeRate | sublist) 
                  + (0 + unsafeSol | sublist),
                  data = train,
                  REML=F)
display(MLexamp22)
summary(MLexamp22)
r.squaredGLMM(MLexamp22)
#R2m       R2c
#[1,] 0.4980643 0.6101207
pred11 = predict(MLexamp22, test)
# scatter plot
plot(pred11, test$humanlen,
     xlab='Pred log(humanlen)', ylab='True log(humanlen)',
     main='Model Found All MAG LMM Random Intercept', 
     cex=0.3)
lines(test$humanlen, test$humanlen, 
      lwd=2)
mtext(text=paste(c("R2 =", r.squaredGLMM(MLexamp22)[2]), collapse = " "), 
      side=3)

###################### correlation between humanlen and rt #####################
cor(train$humanlen, train$rt, method = "pearson")
cor(train$humanlen, train$rt, method = "spearman")
# scatter plot
png(file="/Users/chloe/Documents/RushHour/state_model/out_model/corr_rt_humanlen.png",
    width=650, height=509, res=100)
plot(train$rt, train$humanlen,
     xlab='log(rt)', ylab='log(humanlen)',
     main='Correlation of rt and humanlen', 
     cex=0.3)
mtext(text=paste(c("Pearson corr = ", 
                   cor(train$humanlen, train$rt, method = "pearson")[1], 
                   "\nSpearman corr = ", 
                   cor(train$humanlen, train$rt, method = "spearman")[1]), 
                   collapse = " "), 
      side=3,
      cex=0.8)
dev.off()

