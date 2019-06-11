library(lme4)
library(MuMIn)
library(ggplot2)
require(lme4)
rm(list=ls())
d <- read.csv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')
# d <- read.csv('/Users/chloe/Desktop/random.csv')
scale(d$mobility, center = TRUE, scale = TRUE)
scale(d$diffoptlen, center = TRUE, scale = TRUE)
scale(d$error_made, center = TRUE, scale = TRUE)
scale(d$initial, center = TRUE, scale = TRUE)
# predictor: mobility, diffoptlen, error_made (prev is an error), initial, (1|subject)
# model_mob <- glmer(error_tomake~mobility+error_made+(1|subject), 
#                   data=d, family=binomial,
#                   control = glmerControl(optimizer = "bobyqa"))
# summary(model_mob)
# model_mob1 <- glmer(error_tomake~mobility+error_made+initial+(1|subject), 
#                     data=d, family=binomial,
#                     control = glmerControl(optimizer = "bobyqa"))
# summary(model_mob1)
# 
# 
# model_diffoptlen <- glmer(error_tomake~diffoptlen+error_made+(1|subject), 
#                            data=d, family=binomial,
#                            control = glmerControl(optimizer = "bobyqa"))
# summary(model_diffoptlen)
# model_diffoptlen1 <- glmer(error_tomake~diffoptlen+error_made+initial+(1|subject), 
#                            data=d, family=binomial,
#                            control = glmerControl(optimizer = "bobyqa"))
# summary(model_diffoptlen1)
# 
# model_mob_dif <- glmer(error_tomake~mobility+diffoptlen+error_made+(1|subject), 
#                        data=d, family=binomial,
#                        control = glmerControl(optimizer = "bobyqa"))
# summary(model_mob_dif)
# model_control <- glmer(error_tomake~(1|subject), 
#                        data=d, family=binomial,
#                        control = glmerControl(optimizer = "bobyqa"))
# summary(model_control)
model_mob_dif1 <- glmer(error_tomake~mobility+diffoptlen
                        +error_made+initial+(1|subject), 
                        data=d, family=binomial,
                        control = glmerControl(optimizer = "bobyqa"))
summary(model_mob_dif1)

model_restart <- glmer(restart ~ mobility+diffoptlen+error_made+initial+(1|subject), 
                        data=d, family=binomial,
                        control = glmerControl(optimizer = "bobyqa"))
summary(model_restart)

model_surrender <- glmer(surrender ~ mobility+diffoptlen+error_made+initial+(1|subject), 
                       data=d, family=binomial,
                       control = glmerControl(optimizer = "bobyqa"))
summary(model_surrender)

library("ggpubr")
cor(x=d$diffoptlen, y=d$error_made, method = "spearman")
cor(x=d$diffoptlen, y=d$error_made, method = "spearman")

# model_test <- glmer(error_tomake~mobility+diffoptlen
#                         +error_made+initial+(1|subject),
#                         data=d, family=binomial,
#                         control = glmerControl(optimizer = "bobyqa"))
# summary(model_test)
