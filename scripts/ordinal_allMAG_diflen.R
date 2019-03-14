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
install.packages("DescTools")


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
library(DescTools)

# preprocess data
train <- read.csv(file="/Users/chloe/Documents/RushHour/state_model/in_data_trials/train.csv")
# summary(train)
# head(train)
plot(ecdf(train$diflen))
quantile(train$diflen, probs = seq(0, 0.95, 0.05))
train$diflen <- Winsorize(train$diflen, probs=c(0, 0.50))

train$diflen <- as.ordered(train$diflen)
plot(ecdf(train$diflen))
max(train$diflen)
min(train$diflen)


############################## static MAG and optlen ###########################
# diflen random intercept
MLexamp <- clmm2(diflen ~ c_incycle
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
              + unsafeSol,
              random = sublist,
              # + (1 | sublist),
              data = train,
              link = 'logistic',
              threshold = 'equidistant',
              Hess=TRUE)
logLik(MLexamp) # -8453.22, after winsorizing -7785.88 (df=29)
summary(MLexamp)
condVar(MLexamp)
alpha(MLexamp)
predict(MLexamp)
get_pred(MLexamp)

# get prediction and plot
pred <- fitted.values(MLexamp)
pred
plot(pred, train$diflen)

ndat <- expand.grid(y = gl(51, 1), x = train$diflen)
expand.grid(y = gl(51, 1), x = train$diflen)
(pmat.clm <- matrix(predict(MLexamp, newdata = ndat), ncol = 51, 
                    byrow = TRUE))
(class.clm <- factor(apply(pmat.clm, 1, which.max)))



################################### interval data ##################################
# preprocess data
train <- read.csv(file="/Users/chloe/Documents/RushHour/state_model/in_data_trials/train.csv")
plot(ecdf(train$diflen))
quantile(train$diflen, probs = seq(0, 0.95, 0.05))
train$diflen <- Winsorize(train$diflen, probs=c(0, 0.95))
train$diflen <- as.ordered(CutQ(train$diflen, breaks=quantile(train$diflen, seq(0, 1, by = 0.05))))
plot(ecdf(train$diflen))
max(train$diflen)
min(train$diflen)
# diflen random intercept
MLexamp <- clmm(diflen ~ c_incycle
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
logLik(MLexamp) # -4743.255 (df=29) int0.1, -5937.408 (df=29) int0.05
summary(MLexamp)
condVar(MLexamp)
alpha(MLexamp)
