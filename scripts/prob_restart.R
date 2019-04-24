library(lme4)
library(MuMIn)
library(ggplot2)
rm(list=ls())
d <- read.csv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')


# print(model_all, corr=FALSE)
pred_all <- predict(model_all, newdata=d, type='response')
pred_all

data_all <- cbind(d, pred_all)

######################### try plotting ###################
library(ggplot2)

model_all <- glm(restart ~ consec_error, 
                   data=d, family=binomial("logit"))
model_all

model_further <- glm(restart ~ consec_error_further, 
                 data=d, family=binomial("logit"))
model_further

model_closer <- glm(restart ~ consec_error_closer, 
                     data=d, family=binomial("logit"))
model_closer

model_cross <- glm(restart ~ consec_error_cross, 
                    data=d, family=binomial("logit"))
model_cross


ggplot(d) +
  geom_point(x=d$consec_error, y=d$restart, 
             col='blue', size=0.8) + 
  stat_smooth(mapping=aes(x=consec_error, y=restart),
              method="glm", 
              method.args=list(family="binomial"),
              se=TRUE, col='blue', fullrange = TRUE) +
  coord_cartesian(ylim=c(0, 1.0)) +
  ggtitle("Probability of Restart by Overall Consecutive Error") +
  xlab("Consecutive Error") + ylab("Probability")

ggplot(d) +
  geom_point(x=d$consec_error_further, y=d$restart, 
             col='red', size=0.8) + 
  stat_smooth(mapping=aes(x=consec_error_further, y=restart),
              method="glm", 
              method.args=list(family="binomial"),
              se=TRUE, col='red', fullrange = TRUE) +
  coord_cartesian(ylim=c(0, 1.0)) +
  ggtitle("Probability of Restart by Consecutive Error Further") +
  xlab("Consecutive Error Further") + ylab("Probability")



ggplot(d) +
  geom_point(x=d$consec_error_closer, y=d$restart, 
             col='green', size=0.8) + 
  stat_smooth(mapping=aes(x=consec_error_closer, y=restart),
              method="glm", 
              method.args=list(family="binomial"),
              se=TRUE, col='green', fullrange = TRUE) +
  coord_cartesian(ylim=c(0, 1.0)) +
  ggtitle("Probability of Restart by Consecutive Error Closer") +
  xlab("Consecutive Error Closer") + ylab("Probability")


ggplot(d) +
  geom_point(x=d$consec_error_cross, y=d$restart, 
             col='orange', size=0.8) + 
  stat_smooth(mapping=aes(x=consec_error_cross, y=restart),
              method="glm", 
              method.args=list(family="binomial"),
              se=TRUE, col='orange', fullrange = TRUE) +
  coord_cartesian(ylim=c(0, 1.0)) +
  ggtitle("Probability of Restart by Consecutive Error Cross") +
  xlab("Consecutive Error Cross") + ylab("Probability")


ggplot(d) +
  stat_smooth(mapping=aes(x=consec_error, y=restart),
              method="glm", 
              method.args=list(family="binomial"),
              se=TRUE, col='blue', fullrange = TRUE) +
  stat_smooth(mapping=aes(x=consec_error_further, y=restart),
              method="glm", 
              method.args=list(family="binomial"),
              se=TRUE, col='red', fullrange = TRUE) +
  stat_smooth(mapping=aes(x=consec_error_closer, y=restart),
              method="glm", 
              method.args=list(family="binomial"),
              se=TRUE, col='green', fullrange = TRUE) +
  stat_smooth(mapping=aes(x=consec_error_cross, y=restart),
              method="glm", 
              method.args=list(family="binomial"),
              se=TRUE, col='orange', fullrange = TRUE) +
  ggtitle("Probability of Restart by Consecutive Errors") +
  xlab("Consecutive Errors") + ylab("Probability")
  




###################
ggplot(d) +
  geom_point(x=d$consec_error, y=d$restart, 
             col='blue', size=0.8) + 
  stat_smooth(mapping=aes(x=consec_error, y=restart),
              method="glm", 
              method.args=list(family="binomial"),
              se=TRUE, col='blue', fullrange = TRUE) + 
  geom_point(x=d$consec_error_further, y=d$restart, 
             col='red', size=0.8) + 
  stat_smooth(mapping=aes(x=consec_error_further, y=restart),
                method="glm", 
                method.args=list(family="binomial"),
                se=TRUE, col='red', fullrange = TRUE) + 
  geom_point(x=d$consec_error_further, y=d$restart, 
             col='green', size=0.8) + 
  stat_smooth(mapping=aes(x=consec_error_closer, y=restart),
                method="glm",
                method.args=list(family="binomial"),
                se=TRUE, col='green', fullrange = TRUE) + 
  coord_cartesian(ylim=c(0, 1.0)) +
  theme(legend.position="none")
  
  figure <- ggarrange(bxp, dp, lp,
                      labels = c("A", "B", "C"),
                      ncol = 2, nrow = 2)
  figure



########################### end try ######################


model_further <- glmer(restart ~ consec_error_further + (1|subject), 
                   data=d, family=binomial, control=glmerControl(optimizer='bobyqa'))
print(model_further, corr=FALSE)
pred_further <- predict(model_further, newdata=d, type='response')
pred_further
model_closer <- glmer(restart ~ consec_error_closer + (1|subject), 
                       data=d, family=binomial, control=glmerControl(optimizer='bobyqa'))
print(model_closer, corr=FALSE)
pred_closer <- predict(model_closer, newdata=d, type='response')
pred_closer
plot(d$consec_error, pred_all, type='l',
     xlab='Consecutive errors', ylab='Probability restart',
     main='Prob of restart as function of consecutive error', 
     lwd=0.5, col=alpha('blue', 0.2))
lines(d$consec_error_further, pred_further, type='l',
     lwd=0.5, col=alpha('red', 0.5))
lines(d$consec_error_closer, pred_closer, type='l',
     lwd=0.5, col=alpha('green', 0.8))


