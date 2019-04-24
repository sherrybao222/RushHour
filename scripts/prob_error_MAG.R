library(lme4)
library(MuMIn)
library(ggplot2)
rm(list=ls())
d <- read.csv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')

model <- glmer(error ~ node_human_static+
               edge_human_static+
               en_human_static+
               enp_human_static+
               e2n_human_static+
               scc_human_static+
               maxscc_human_static+
               cycle_human_static+
               maxcycle_human_static+
               cincycle_human_static+
               ninc_human_static+
               pnc_human_static+
               depth_human_static+
               ndepth_human_static+
               gcluster_human_static+
               lcluster_human_static+ (1|subject), 
                data=d, family=binomial("logit"))
model

