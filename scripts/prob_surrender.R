library(lme4)
library(MuMIn)
library(ggplot2)
rm(list=ls())
d <- read.csv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')

######################### try plotting ###################

model_all <- glm(surrender ~ consec_error, 
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

library(ggplot2)

ggplot(d) +
  geom_point(x=d$consec_error, y=d$surrender, 
             col='blue', size=0.8) + 
  stat_smooth(mapping=aes(x=consec_error, y=surrender),
              method="glm", 
              method.args=list(family="binomial"),
              se=TRUE, col='blue', fullrange = TRUE) +
  coord_cartesian(ylim=c(0, 1.0)) +
  ggtitle("Probability of Surrender by Overall Consecutive Error") +
  xlab("Consecutive Error") + ylab("Probability")

ggplot(d) +
  geom_point(x=d$consec_error_further, y=d$surrender, 
             col='red', size=0.8) + 
  stat_smooth(mapping=aes(x=consec_error_further, y=surrender),
              method="glm", 
              method.args=list(family="binomial"),
              se=TRUE, col='red', fullrange = TRUE) +
  coord_cartesian(ylim=c(0, 1.0)) +
  ggtitle("Probability of surrender by Consecutive Error Further") +
  xlab("Consecutive Error Further") + ylab("Probability")



ggplot(d) +
  geom_point(x=d$consec_error_closer, y=d$surrender, 
             col='green', size=0.8) + 
  stat_smooth(mapping=aes(x=consec_error_closer, y=surrender),
              method="glm", 
              method.args=list(family="binomial"),
              se=TRUE, col='green', fullrange = TRUE) +
  coord_cartesian(ylim=c(0, 1.0)) +
  ggtitle("Probability of surrender by Consecutive Error Closer") +
  xlab("Consecutive Error Closer") + ylab("Probability")


ggplot(d) +
  geom_point(x=d$consec_error_cross, y=d$surrender, 
             col='orange', size=0.8) + 
  stat_smooth(mapping=aes(x=consec_error_cross, y=surrender),
              method="glm", 
              method.args=list(family="binomial"),
              se=TRUE, col='orange', fullrange = TRUE) +
  coord_cartesian(ylim=c(0, 1.0)) +
  ggtitle("Probability of Surrender by Consecutive Error Cross") +
  xlab("Consecutive Error Cross") + ylab("Probability")


ggplot(d) +
  stat_smooth(mapping=aes(x=consec_error, y=surrender),
              method="glm", 
              method.args=list(family="binomial"),
              se=TRUE, col='blue', fullrange = TRUE) +
  stat_smooth(mapping=aes(x=consec_error_further, y=surrender),
              method="glm", 
              method.args=list(family="binomial"),
              se=TRUE, col='red', fullrange = TRUE) +
  stat_smooth(mapping=aes(x=consec_error_closer, y=surrender),
              method="glm", 
              method.args=list(family="binomial"),
              se=TRUE, col='green', fullrange = TRUE) +
  stat_smooth(mapping=aes(x=consec_error_cross, y=surrender),
              method="glm", 
              method.args=list(family="binomial"),
              se=TRUE, col='orange', fullrange = TRUE) +
  ggtitle("Probability of Surrender by Consecutive Errors") +
  xlab("Consecutive Errors") + ylab("Probability")

