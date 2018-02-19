#Analysis and comparison of the LGHOO with the HOO algorithm
#Author: David Issa Mattos
#Email: davidis@chalmers.se

#Cleaning the space
rm(list=ls())

library(ggplot2)
library(psych)
library(car)
library(compute.es)
library(multcomp)
library(coin)

#Loading the data

#LGHOO with minimum grow of 20 with best arm selection
lghoo.mingrow.20.bestarm <- read.csv('montecarlo_randomPoly-numsim1000mingrow20arm_policy-newhorizon-1000.csv', header = TRUE)
lghoo.mingrow.20.bestarm$group <- "LGHOO"

#HOO original
hoo.original <- read.csv('montecarlo_randomPoly-numsim1000mingrow0arm_policy-originalhorizon-1000.csv', header = TRUE)
hoo.original$group <- "HOO"

#Testing for normality
shapiro.test(lghoo.mingrow.20.bestarm$euclidian_distance)
shapiro.test(lghoo.mingrow.20.bestarm$time_spent)
shapiro.test(hoo.original$euclidian_distance)
shapiro.test(hoo.original$time_spent)

expdata <- rbind(lghoo.mingrow.20.bestarm,hoo.original)
expdata$group <- as.factor(expdata$group)
#Some summary statistics
print("Euclidian distance:")
describeBy(expdata$euclidian_distance, group = expdata$group)
print("Time spent:")
describeBy(expdata$time_spent, group = expdata$group)

ggplot( data= expdata, aes(x=euclidian_distance, fill = group)) + 
  geom_histogram(alpha=0.5, bins = 20)+
  ggtitle("Histogram for the euclidian distance")
ggplot( data= expdata, aes(x=time_spent, fill = group)) + 
  geom_histogram(alpha=0.5, bins = 20)+
  ggtitle("Histogram for the time spent")


#As the data is not normal
#Testing for the euclidian distance
wilcox.test(euclidian_distance ~ group, data = expdata, conf.int=TRUE, conf.level=0.95, alternative = "two.sided")

#Testing for the time
wilcox.test(time_spent ~ group, data = expdata, conf.int=TRUE, conf.level=0.95, alternative = "two.sided")


