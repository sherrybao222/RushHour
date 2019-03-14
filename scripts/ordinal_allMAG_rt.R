# Trial by trial prediction using Ordinal Regression Mixed Model
# Predictors are dynamic MAG and static MAG, and optlen as well
# predict humane solution length - optimal solution length

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
install.packages("ordinal")


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
library(ordinal)
train <- read.csv(file="/Users/chloe/Documents/RushHour/state_model/in_data_trials/train.csv")
summary(train)
summary(test)
head(train)
head(test)

# preprocess data
train$rt <- cut(train$rt, breaks=15)

############################## static MAG and optlen ###########################
# diflen random intercept
MLexamp <- clmm(rt ~ c_incycle
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
                data = train,
                link = 'logit',
                threshold = 'equidistant')
summary(MLexamp)
logLik(MLexamp) # -1002.098 (df=29)
condVar(MLexamp)
alpha(MLexamp)
pred <- fitted.values(MLexamp)
pred
plot(pred, train$rt)


