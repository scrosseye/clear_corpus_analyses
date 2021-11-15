## This document performs a linear model to predict readability using language
#features derived from SALAT (see linguisticanalysistools.org)

################################################################################
# Load Libraries
################################################################################

# If you don't have these libraries installed, you will have to intall them
library(caret)#machine learning
library(dplyr)#data analytics
library(psych)#descriptives
library(tidyverse)#data analytics
library(leaps)#stepwise
library(car)#vif
library(relaimpo)#variable importance
library(Hmisc)#correlations

################################################################################
# clear previous data and max printing
################################################################################


rm(list=ls(all=TRUE)) #clear memory
options(max.print=1000000)


################################################################################
# Prepare Data
################################################################################

# read in file that contains all variables with high non-zero counts
data <- read.csv(file = 'final_clear_variables_for_analysis_nlp_meta.csv')


# examine data
str(data)

#describe the data
psych::describe(data)


# scale linguistic data 
data1 <- data %>%
  mutate_at(c(26:426), funs(c(scale(.))))

# ensure data is mean centered
psych::describe(data1)

#select only the variables that are not multicollinear including the DV. 
#this is from a previous analysis that pruned variables based on multicollinearity
sel_data <- readLines("final_variables_for_lm.csv")

data_source <- data1[, sel_data] #non-MC variables

#examine data
str(data_source)
psych::describe(data_source)

################################################################################
# Correlation analysis
################################################################################

#correlateion
cor(data_source)

#correlation matrix with p values (requires Hmisc)
correl2 <- rcorr(as.matrix(data_source)) 
correl2

# Extract the correlation coefficients
corr_r <- correl2$r

#round them to 3 decimals
corr_r_round <- round(corr_r, 3)
corr_r_round

#write to csv
write.csv(write.csv(corr_r_round,"corr_matrix_non_mc_variables_bt_ease.csv"))

################################################################################
# Cross-validated LM
################################################################################

#a long process is skipped here that involved hand pruning of variable that showed
#suppression effects. Initially, a stepwise model was run with 107 variables in 
#the data_source DF. The first model included 50 variables, but many of these showed
#suppression effects (i.e., the correlation in corr_r_round was positive, but the 
#co-efficient was negative). For each model, the first variable that showed suppression
#effects was removed and a new model was run. In the end, 28 variables were kept
#for analysis. These are the variables below (sel_data2)


sel_data2 <- c("BT_easiness",
               "SUBTLEXus_Range_CW_Log",
               "Sem_D",
               "MRC_Imageability_CW",
               "basic_ntypes",
               "LD_Mean_Accuracy",
               "fic_lemma_construction_attested", 
               "Complet_GI",
               "Timespc_Lasswell",
               "all_collexeme_ratio_type",
               "OG_N_H",
               "adjacent_overlap_argument_sent",
               "SUBTLEXus_Range_FW",
               "av_nsubj_deps_NN",
               "Sv_GI",
               "lsa_average_top_three_cosine",
               "Socrel_GI",
               "nsubj_per_cl",
               "aoe_inverse_linear_regression_slope",
               "Anticipation_EmoLex",
               "USF_FW",
               "fic_collexeme_ratio",
               "Valence_nwords", 
               "Brysbaert_Concreteness_Combined_AW",
               "all_temporal",
               "Role_GI",
               "OLDF_FW",
               "news_construction_attested",
               "WN_SD_CW")

#select only the variable that do not show suppression effects
data_source3 <- data_source[, sel_data2] #variables from LM model
str(data_source3)

#set seed for replication of cross-validation at later time
set.seed(123)

# Set up repeated k-fold cross-validation
train.control <- trainControl(method = "cv", number = 10)

#the LM stepwise model used
step.model.source <- train(BT_easiness ~ ., data = data_source3,
                           method = "leapSeq", #stepwise selection 
                           tuneGrid = data.frame(nvmax = 1:28), #using 1-28 predictors
                           trControl = train.control)

#the model
step.model.source$results

#best tuned model
step.model.source$bestTune

#which variables were strong predictors
summary(step.model.source$finalModel)

#co-efficients for model using all 28 variables
coef(step.model.source$finalModel, 28)

#variable importance
varImp(step.model.source)$importance


#conducted a final LM model after stepwise above in order measure F value, t values
#VIF, residuals, homoscedasticity, and variable importance

final <- lm(formula = BT_easiness ~ SUBTLEXus_Range_CW_Log + Sem_D + MRC_Imageability_CW + 
              basic_ntypes + LD_Mean_Accuracy + fic_lemma_construction_attested + 
              Complet_GI + Timespc_Lasswell + all_collexeme_ratio_type + 
              OG_N_H + adjacent_overlap_argument_sent + SUBTLEXus_Range_FW + 
              av_nsubj_deps_NN + Sv_GI + lsa_average_top_three_cosine + 
              Socrel_GI + nsubj_per_cl + aoe_inverse_linear_regression_slope + 
              Anticipation_EmoLex + USF_FW + fic_collexeme_ratio + Valence_nwords + 
              Brysbaert_Concreteness_Combined_AW + all_temporal + Role_GI + 
              OLDF_FW + news_construction_attested + WN_SD_CW, 
              data = data_source)

#summary of model with additional statistics
summary (final)

#calculate variable importance metrics. This will likely require a high-performance machine
metrics_w_types_all <- calc.relimp(final)#, type = c("lmg", "pmvd", "first", "last", "betasq", "pratt"))
metrics_w_types_all

car::vif(final) #VIF values for the regression to ensure no problems with multi-collinearity

#normality
shapiro.test(residuals(final)) #residual plot to check for normality

#Homoscedasticity check
plot(final, which = 1) #residual plot


