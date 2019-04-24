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
train <- read.csv(file="/Users/chloe/Documents/RushHour/exp_data/moves_MAG1000.csv")
summary(train)
head(train)


############################## static MAG and optlen ###########################
# rt random intercept
MLexamp <- glmer(restart ~ p_unsafe_sol
                 + p_backmove_sol
                 + avg_node_sol
                 + avg_edge_sol	
                 + avg_ncycle_sol	
                 + avg_maxcycle_sol	
                 + avg_node_incycle_sol	
                 + avg_depth_sol	
                 + node_rate	
                 + edge_rate
                + (1 | worker),
                family=binomial(link = "logit"),
                data = train)
display(MLexamp)
summary(MLexamp)