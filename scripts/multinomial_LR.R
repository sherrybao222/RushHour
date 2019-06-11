require(foreign)
require(nnet)
require(ggplot2)
require(reshape2)
rm(list=ls())
d <- read.csv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')

test <- multinom(decision~mobility+diffoptlen+initial, data = d)
summary(test)
