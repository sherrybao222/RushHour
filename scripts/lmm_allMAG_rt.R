# Trial by trial prediction using Linear Mixed Model
# Predictors are dynamic MAG and static MAG, and optlen as well
# predict response time

# install library
install.packages("lme4")
install.packages("arm")
install.packages("sjmisc")
install.packages("MuMIn")
install.packages("lmehumanlenest")
install.packages("optimx")
install.packages("viridis")
install.packages("r2glmm")
install.packages("ppcor")


# initialize
rm(list = ls())
library(lme4)  # load library
library(nlme)
library(ppcor)
library(arm)  # convenience functions for regression in R
library(sjmisc)
library(MuMIn)
library(lmertest)
library(optimx)
library(ggplot2)
library(viridis)
library(r2glmm)
train <- read.csv(file="/Users/chloe/Documents/RushHour/state_model/in_data_trials/train.csv")
test <- read.csv(file="/Users/chloe/Documents/RushHour/state_model/in_data_trials/test.csv")
summary(train)
summary(test)
head(train)
head(test)

# preprocess data
train$rt <- log(train$rt)
test$rt <- log(test$rt)

############################## static MAG and optlen ###########################
# rt random intercept
MLexamp <- lmer(rt ~ c_incycle
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
# R2m       R2c
# [1,] 0.392747 0.5364988
r2beta(MLexamp, partial=TRUE)
pred = predict(MLexamp, test)
temp <- cbind(test$rt, pred)
temp
pcor(temp)
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
MLexamp11 <- lmer(rt ~ c_incycle 
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
                  data = train,
                  REML=F)
display(MLexamp11)
summary(MLexamp11)
r.squaredGLMM(MLexamp11)
#R2m       R2c
#[1,] 0.3927591 0.5349559
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
MLexamp2 <- lmer(rt ~ c_incycle + (0 + c_incycle | sublist)
                  + countcycle + (0 + countcycle | sublist)
                  + countscc + (0 + countscc | sublist)
                  + depth + (0 + depth | sublist)
                  + e2n + (0 + e2n | sublist)
                  + edges + (0 + edges | sublist)
                  + maxcycle + (0 + maxcycle | sublist)
                  + maxscc + (0 + maxscc | sublist)
                  + ncc + (0 + ncc | sublist)
                  + pnc + (0 + pnc | sublist)
                  + avgDepthSol + (0 + avgDepthSol | sublist)
                  + avgEdgeSol + (0 + avgEdgeSol | sublist)
                  + nodeRate + (0 + nodeRate | sublist)
                  + (1 | sublist),
                  data = train,
                  REML=F)
display(MLexamp2)
summary(MLexamp2)
r.squaredGLMM(MLexamp2)
# R2m       R2c
# [1,] 0.3643591 0.5859223
pred2 = predict(MLexamp2, test)
# stepwise regression
step_result <- step(MLexamp2)
step_result
# MLexamp22 <- lmer(humanlen ~ c_incycle 
#                   + countcycle 
#                   + countscc 
#                   + depth 
#                   + e2n 
#                   + edges 
#                   + en 
#                   + gcluster + (0 + gcluster | sublist)
#                   + lcluster 
#                   + maxcycle 
#                   + maxscc 
#                   + ncc 
#                   + nodes + (0 + nodes | sublist)
#                   + pnc 
#                   + optlen + (0 + optlen | sublist)
#                   + avgDepthSol + (0 + avgDepthSol | sublist)
#                   + avgEdgeSol + (0 + avgEdgeSol | sublist)
#                   + avgMaxCycleSol + (0 + avgMaxCycleSol | sublist)
#                   + nodeRate + (0 + nodeRate | sublist),
#                   data = train,
#                   REML=F)
# display(MLexamp22)
# summary(MLexamp22)
# r.squaredGLMM(MLexamp22)
# # R2m       R2c
# # [1,] 0.5052556 0.6037029
# pred22 = predict(MLexamp22, test)




################################### PLOT BEST MODE #################################
# prepare data
best_model <- MLexamp
plot_dir1 <- "/Users/chloe/Documents/RushHour/state_model/out_model/slope-rt-ellipse-CI068-puz.png"
plot_dir2 <- "/Users/chloe/Documents/RushHour/state_model/out_model/slope-rt-ellipse-CI068-optlen.png"
plot_dir3 <- "/Users/chloe/Documents/RushHour/state_model/out_model/slope-rt-scatter-puz.png"
plot_dir4 <- "/Users/chloe/Documents/RushHour/state_model/out_model/slope-rt-scatter-optlen.png"
best_title <- "All MAG + optlen -> rt, LMM random int"
pred_frame <- data.frame(pred)
pred_frame
test_frame <- data.frame(test)
test_frame
ggdat <- cbind(test_frame, pred_frame)
ggdat$optlen <- as.character(as.numeric(ggdat$optlen))
class(ggdat$optlen)
ggdat$optlen
# ggplot ellipse by puzzle
p <- ggplot(ggdat, aes(pred, rt, colour=puzzle)) + 
  geom_point(size=1, alpha=0.6, show.legend=FALSE) + 
  labs(x = "Pred log(rt)") + 
  labs(y = "True log(rt)") + 
  labs(title = best_title, subtitle = paste(c("R2 =", r.squaredGLMM(best_model)[2]), collapse = " ")) + 
  geom_line(aes(pred, pred), alpha=0.5,show.legend = FALSE) +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5)) + 
  labs(caption = "coloured by puzzle") + 
  theme(panel.grid.minor=element_line(colour="grey", linetype=3, size=0.5)) +
  # scale_y_continuous(minor_breaks = c(log(7),log(11),log(14),log(16)), breaks = seq(0,6,0.5)) +
  # scale_x_continuous(minor_breaks = c(log(7),log(11),log(14),log(16)), breaks = seq(0,5,0.5)) +
  stat_ellipse(level=0.68, alpha=0.4, type = "norm", show.legend = FALSE)
p
ggsave(plot_dir1)
# ggplot ellipse group by puzzle, color by optlen
p <- ggplot(ggdat, aes(pred, rt, group=puzzle, colour=optlen)) + 
  geom_point(size=1, alpha=0.6, aes(group=puzzle, colour=optlen)) + 
  labs(x = "Pred log(rt)") + 
  labs(y = "True log(rt)") + 
  labs(title = best_title, subtitle = paste(c("R2 =", r.squaredGLMM(best_model)[2]), collapse = " ")) + 
  geom_line(aes(pred, pred), alpha=0.5,show.legend = FALSE) +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5)) + 
  labs(caption = "grouped by puzzle, coloured by optlen") + 
  theme(panel.grid.minor=element_line(colour="grey", linetype=3, size=0.5)) +
  # scale_y_continuous(minor_breaks = c(log(7),log(11),log(14),log(16)), breaks = seq(0,6,0.5)) +
  # scale_x_continuous(minor_breaks = c(log(7),log(11),log(14),log(16)), breaks = seq(0,5,0.5)) +
  stat_ellipse(level=0.68, alpha=0.4, type = "norm") + 
  scale_colour_brewer(palette = "Set2", breaks = c("16","14","11","7"))
p
ggsave(plot_dir2)
# ggplot scatter by puzzle
p <- ggplot(ggdat, aes(pred, rt, colour=puzzle)) + 
  geom_point(size=1, alpha=0.95, show.legend=FALSE) + labs(x = "Pred log(rt)") + 
  labs(y = "True log(rt)") +
  labs(title = best_title, subtitle = paste(c("R2 =", r.squaredGLMM(best_model)[2]), collapse = " ")) + 
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5)) + 
  labs(caption = "coloured by puzzle") +
  geom_line(aes(pred, pred), show.legend = FALSE) +
  theme(panel.grid.minor=element_line(colour="grey", linetype=3, size=0.5)) 
# scale_y_continuous(minor_breaks = c(log(7),log(11),log(14),log(16)), breaks = seq(0,6,0.5)) +
# scale_x_continuous(minor_breaks = c(log(7),log(11),log(14),log(16)), breaks = seq(0,5,0.5))
p
ggsave(plot_dir3)
# ggplot scatter by optlen
p <- ggplot(ggdat, aes(pred, rt)) +
  geom_point(size=1, alpha=0.95, aes(group=optlen, colour=optlen)) + 
  scale_colour_brewer(palette = "Set2", breaks = c("16","14","11","7")) + 
  labs(x = "Pred log(rt)") + 
  labs(y = "True log(rt)") + 
  labs(title = best_title, subtitle = paste(c("R2 =", r.squaredGLMM(best_model)[2]), collapse = " ")) +
  labs(caption = "coloured by optlen") +
  theme(plot.title=element_text(hjust=0.5), plot.subtitle=element_text(hjust = 0.5)) +
  theme(panel.grid.minor=element_line(colour="grey", linetype=3, size=0.5)) +
  # scale_y_continuous(minor_breaks = c(log(7),log(11),log(14),log(16)), breaks = seq(0,6,0.5)) +
  # scale_x_continuous(minor_breaks = c(log(7),log(11),log(14),log(16)), breaks = seq(0,5,0.5)) + 
  geom_line(aes(pred, pred, colour=optlen), show.legend = FALSE, size = 1)
p
ggsave(plot_dir4)