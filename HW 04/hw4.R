############################################################
## Synthetic Visa Prediction Challenge
## Run this whole script in RStudio
############################################################

## (Install once if needed)
## install.packages("rpart")
## install.packages("rpart.plot")
## install.packages("randomForest")

library(rpart)        # decision tree
library(rpart.plot)   # plot decision tree
library(randomForest) # random forest model

set.seed(123)  # for reproducibility of the whole dataset

############################################################
## 1. Generate base features (schema)
##    -> build all the columns that describe each applicant
############################################################

n <- 8000  # number of applicants (rows)

## Age: 18–35 (integer) – basic demographic feature
Age <- sample(18:35, size = n, replace = TRUE)

## Gender – categorical variable
Gender <- sample(c("Female", "Male", "Other"),
                 size = n, replace = TRUE,
                 prob = c(0.48, 0.48, 0.04))

## Region (NOISE – not used in rules; should not matter much for label)
Region <- sample(c("East_Asia", "South_Asia", "Europe",
                   "Latin_America", "Africa", "Middle_East"),
                 size = n, replace = TRUE,
                 prob = c(0.27, 0.20, 0.18, 0.12, 0.13, 0.10))

## Family_Income: skewed (log-normal style), around 10k–200k
## This simulates realistic income distribution (many low, few high).
Family_Income <- round(10^(rnorm(n, mean = 4.7, sd = 0.45)))

## Program_Level – which degree level the student applies for
Program_Level <- sample(c("Undergrad", "Masters", "PhD"),
                        size = n, replace = TRUE,
                        prob = c(0.55, 0.30, 0.15))

## Field_of_Study – broad area of study
Field_of_Study <- sample(c("STEM", "Business", "Social_Science", "Arts"),
                         size = n, replace = TRUE,
                         prob = c(0.35, 0.30, 0.20, 0.15))

## GPA: normal distribution, truncated to [2.0, 4.0]
## This is one of the key “academic strength” features.
GPA <- rnorm(n, mean = 3.2, sd = 0.4)
GPA <- pmax(pmin(GPA, 4.0), 2.0)

## IELTS_Score: 4.0–9.0, step 0.5
## Another academic/English proficiency feature.
IELTS_Score <- sample(seq(4.0, 9.0, by = 0.5),
                      size = n, replace = TRUE,
                      prob = c(0.02, 0.03, 0.05, 0.08, 0.15,
                               0.20, 0.18, 0.12, 0.10, 0.05, 0.02))

## Num_Countries_Visited: Poisson, capped at 15
## Approximates travel history / global exposure.
Num_Countries_Visited <- rpois(n, lambda = 2)
Num_Countries_Visited[Num_Countries_Visited > 15] <- 15

## Funding_Type – how the studies are funded
Funding_Type <- sample(c("Self", "Family", "Scholarship", "Mixed"),
                       size = n, replace = TRUE,
                       prob = c(0.30, 0.35, 0.15, 0.20))

## Has_sibling: Poisson, capped at 5
## Simple family structure variable.
Has_sibling <- rpois(n, lambda = 2)
Has_sibling[Has_sibling > 5] <- 5

## Has_US_Relative – important for mobility / ties in rules
Has_US_Relative <- sample(c("Yes", "No"),
                          size = n, replace = TRUE,
                          prob = c(0.30, 0.70))

## Season_Applied – timing of application
Season_Applied <- sample(c("Fall", "Spring", "Summer"),
                         size = n, replace = TRUE,
                         prob = c(0.50, 0.30, 0.20))

## Num_Social_Media_Platforms (NOISE – not used in rules)
## This is intentionally noise so good models should ignore it.
Num_Social_Media_Platforms <- pmin(rpois(n, lambda = 4), 10)

## Make everything into a data frame (this is the main dataset)
df <- data.frame(
  Age,
  Gender,
  Region,
  Family_Income,
  Program_Level,
  Field_of_Study,
  GPA,
  IELTS_Score,
  Num_Countries_Visited,
  Funding_Type,
  Has_sibling,
  Has_US_Relative,
  Season_Applied,
  Num_Social_Media_Platforms,
  stringsAsFactors = FALSE
)

############################################################
## 2. Hidden score + Visa_Approved (hidden rules)
##    -> build an internal “score” and convert it to 0/1 label
############################################################

## 2.1. Shared intermediate variables (used across all programs)

## academic_raw: combined effect of GPA and IELTS
academic_raw <- 0.6 * (df$GPA - 3.0) +
  0.4 * ((df$IELTS_Score - 6.5) / 1.5)

## income_log and income_centered: log income, centered around 10^4.7 (~50k)
income_log      <- log10(df$Family_Income + 1)
income_centered <- income_log - 4.7   # ≈ 0 for income ~ 50k

## mobility_raw: more countries + US relative = more mobility score
mobility_raw <- 0.10 * df$Num_Countries_Visited +
  0.60 * (df$Has_US_Relative == "Yes")

## funding_bonus: scholarship > mixed > family > self
funding_bonus <- ifelse(df$Funding_Type == "Scholarship", 1.0,
                        ifelse(df$Funding_Type == "Mixed",      0.4,
                               ifelse(df$Funding_Type == "Family",     0.3, 0.0)))

## season_effect: Fall best, Summer worst
season_effect <- ifelse(df$Season_Applied == "Fall",   0.30,
                        ifelse(df$Season_Applied == "Spring", 0.00, -0.30))

## CHANGED: make age_penalty weaker so the pattern is gentler, easier to learn
## Idea: “weird” age for program level slightly hurts the score.
age_penalty <- ifelse(df$Program_Level == "Undergrad" & df$Age > 28, -0.25,
                      ifelse(df$Program_Level == "Masters"   & df$Age < 23, -0.25,
                             ifelse(df$Program_Level == "PhD"       & df$Age < 25, -0.25, 0.0)))

## 2.2. Program-specific rules – different scoring formula by Program_Level

score <- numeric(nrow(df))  # initialize score vector

idx_u <- df$Program_Level == "Undergrad"
idx_m <- df$Program_Level == "Masters"
idx_p <- df$Program_Level == "PhD"

## Undergrad – simpler rules, focus on academics + funding + mobility
score[idx_u] <-
  -0.20 +
  1.20 * academic_raw[idx_u] +
  0.50 * funding_bonus[idx_u] +
  0.20 * mobility_raw[idx_u] -
  0.30 * (df$GPA[idx_u] < 2.8)  # penalty if GPA < 2.8

## Masters – stronger weight on academics + income + mobility
score[idx_m] <-
  -0.50 +
  1.40 * academic_raw[idx_m] +
  0.30 * income_centered[idx_m] +
  0.25 * mobility_raw[idx_m] +
  0.30 * (df$Field_of_Study[idx_m] %in% c("STEM", "Business"))

## PhD – CHANGED: removed ^2 to keep it linear in academic_raw
## Emphasize academics, STEM, funding, and mobility
score[idx_p] <-
  -0.60 +                                 # softer intercept
  1.70 * academic_raw[idx_p] +            # strong academic weight
  0.35 * income_centered[idx_p] +
  0.30 * mobility_raw[idx_p] +
  0.70 * (df$Field_of_Study[idx_p] == "STEM") +
  0.35 * (df$Funding_Type[idx_p] == "Scholarship") -
  0.20 * (df$Has_sibling[idx_p] >= 3)

## 2.3. Add season + age effects to the final score
score <- score + season_effect + age_penalty

## 2.4. Create binary label from score, then flip 2% to add label noise
## This sets an approximate upper bound of ~98% accuracy for any model.
raw_label <- ifelse(score > 0, 1L, 0L)
flip      <- rbinom(nrow(df), size = 1, prob = 0.02)
Visa_Approved <- ifelse(flip == 1L, 1L - raw_label, raw_label)

df$Visa_Approved <- Visa_Approved  # attach target to dataset

############################################################
## 2. Plots (relationship exploration)
##    NOTE: this section uses 'train' later; it is meant to
##    be run after the train/test split or with df as train.
############################################################

## Boxplot GPA & IELTS by Visa_Approved
par(mfrow = c(1, 2))  # show two plots side by side

boxplot(GPA ~ Visa_Approved, data = train,
        names = c("Denied (0)", "Approved (1)"),
        main = "GPA vs Visa_Approved",
        ylab = "GPA")

boxplot(IELTS_Score ~ Visa_Approved, data = train,
        names = c("Denied (0)", "Approved (1)"),
        main = "IELTS vs Visa_Approved",
        ylab = "IELTS Score")

par(mfrow = c(1, 1))  # reset layout

############################################################
## 2. Additional plots
##    -> check if the hidden rules appear in simple summaries
############################################################

## Academic_Index is very similar to academic_raw (no scaling /1.5)
train$Academic_Index <- 0.6 * (train$GPA - 3.0) +
  0.4 * (train$IELTS_Score - 6.5)

boxplot(Academic_Index ~ Visa_Approved, data = train,
        names = c("Denied (0)", "Approved (1)"),
        main = "Academic_Index vs Visa_Approved",
        ylab = "Academic_Index")

## 1) Create visa_num = 0/1 numeric version of Visa_Approved
if (is.factor(train$Visa_Approved)) {
  visa_num <- as.numeric(as.character(train$Visa_Approved))
} else {
  visa_num <- train$Visa_Approved
}

## 2) Define approval_rate(): mean of visa_num within each group
approval_rate <- function(group) {
  tapply(visa_num, group, mean)
}

## Visa approval rate by Program_Level – should reflect rules
vr_program <- approval_rate(train$Program_Level)

barplot(vr_program,
        main = "Visa approval rate by Program_Level",
        ylab = "Approval rate",
        ylim = c(0, 1))

## Visa approval rate by Funding_Type – Scholarship should be highest
vr_funding <- approval_rate(train$Funding_Type)

barplot(vr_funding,
        main = "Visa approval rate by Funding_Type",
        ylab = "Approval rate",
        ylim = c(0, 1))

## Visa approval rate by Season_Applied – Fall > Spring > Summer
vr_season <- approval_rate(train$Season_Applied)

barplot(vr_season,
        main = "Visa approval rate by Season_Applied",
        ylab = "Approval rate",
        ylim = c(0, 1))

## Visa approval rate by Has_US_Relative – “Yes” should be higher
vr_relative <- approval_rate(train$Has_US_Relative)

barplot(vr_relative,
        main = "Visa approval rate by Has_US_Relative",
        ylab = "Approval rate",
        ylim = c(0, 1))

## Mean Visa_Approved by number of countries visited
## We expect approval to slowly increase with more travel.
vr_countries <- approval_rate(train$Num_Countries_Visited)

plot(as.numeric(names(vr_countries)), vr_countries, type = "b",
     main = "Approval rate by Num_Countries_Visited",
     xlab = "Num_Countries_Visited",
     ylab = "Approval rate",
     ylim = c(0, 1))

## Log-transformed income – easier to see relationship with label
train$Log_Income <- log10(train$Family_Income + 1)

boxplot(Log_Income ~ Visa_Approved, data = train,
        names = c("Denied (0)", "Approved (1)"),
        main = "Log_Income vs Visa_Approved",
        ylab = "log10(Family_Income+1)")

## Filter PhD only to confirm special PhD rules
phd <- subset(train, Program_Level == "PhD")
phd_visa <- if (is.factor(phd$Visa_Approved)) as.numeric(as.character(phd$Visa_Approved)) else phd$Visa_Approved

tapply(phd_visa, phd$Field_of_Study, mean)    # STEM vs others for PhD
tapply(phd_visa, phd$Funding_Type, mean)      # Scholarship vs others for PhD

## Undergrad only: check low-GPA penalty
ug <- subset(train, Program_Level == "Undergrad")
ug_visa <- if (is.factor(ug$Visa_Approved)) as.numeric(as.character(ug$Visa_Approved)) else ug$Visa_Approved

ug$Low_GPA <- ug$GPA < 2.8
tapply(ug_visa, ug$Low_GPA, mean)

## Age mismatch indicator for plotting age_penalty effect
train$Age_Mismatch <- as.integer(
  (train$Program_Level == "Undergrad" & train$Age > 28) |
    (train$Program_Level == "Masters"   & train$Age < 23) |
    (train$Program_Level == "PhD"       & train$Age < 25)
)

vr_age_mismatch <- approval_rate(train$Age_Mismatch)

barplot(vr_age_mismatch,
        names.arg = c("No mismatch", "Mismatch"),
        main = "Visa approval rate by Age_Mismatch",
        ylab = "Approval rate",
        ylim = c(0, 1))

## Region (should behave like noise – if strong pattern appears, something is off)
vr_region <- approval_rate(train$Region)
barplot(vr_region,
        main = "Visa approval rate by Region (noise)",
        ylab = "Approval rate",
        ylim = c(0, 1))

## Num_Social_Media_Platforms as pure noise
## We expect almost zero correlation with visa approval.
plot(jitter(train$Num_Social_Media_Platforms),
     jitter(visa_num),
     main = "Num_Social_Media_Platforms vs Visa_Approved",
     xlab = "Num_Social_Media_Platforms",
     ylab = "Visa_Approved (0/1)",
     pch = 16, cex = 0.5)
abline(h = mean(visa_num), lty = 2)

cat("\nCorrelation with Num_Social_Media_Platforms:\n")
print(cor(train$Num_Social_Media_Platforms, visa_num))

############################################################
## 3. Train / Test split (80% / 20%)
##    -> split df into training and testing sets
############################################################

set.seed(456)  # new seed for splitting
train_idx <- sample(seq_len(nrow(df)), size = round(0.8 * nrow(df)))
train <- df[train_idx, ]
test  <- df[-train_idx, ]

## (Optional) save files for the challenge
write.csv(train, "visa_train.csv", row.names = FALSE)
write.csv(test,  "visa_test.csv",  row.names = FALSE)

## Save numeric copies for solution model & oracle (we will engineer features here)
train_num <- train
test_num  <- test

############################################################
## 4. Verification: check distributions & rules
##    -> sanity checks to make sure data generation is correct
############################################################

## IELTS must be spaced by 0.5 (by construction)
cat("Unique IELTS scores:\n")
print(sort(unique(train$IELTS_Score)))

## Basic summaries for key numeric variables
cat("\nSummary GPA:\n")
print(summary(train$GPA))

cat("\nSummary Family_Income:\n")
print(summary(train$Family_Income))

## Check counts of categorical variables
cat("\nProgram_Level counts:\n")
print(table(train$Program_Level))

cat("\nField_of_Study counts:\n")
print(table(train$Field_of_Study))

cat("\nSeason_Applied counts:\n")
print(table(train$Season_Applied))

cat("\nFunding_Type counts:\n")
print(table(train$Funding_Type))

## Overall visa approval rate – helps to see class balance
cat("\nOverall Visa approval rate (train):\n")
print(mean(train$Visa_Approved))

## Approval rate by main variables – should reflect hidden rules
cat("\nVisa rate by Program_Level:\n")
print(tapply(train$Visa_Approved, train$Program_Level, mean))

cat("\nVisa rate by Funding_Type:\n")
print(tapply(train$Visa_Approved, train$Funding_Type, mean))

cat("\nVisa rate by Season_Applied:\n")
print(tapply(train$Visa_Approved, train$Season_Applied, mean))

cat("\nVisa rate by Field_of_Study:\n")
print(tapply(train$Visa_Approved, train$Field_of_Study, mean))

## Check noise columns
cat("\nVisa rate by Region (should be fairly flat / noisy):\n")
print(tapply(train$Visa_Approved, train$Region, mean))

cat("\nCorrelation with Num_Social_Media_Platforms (should be near 0):\n")
print(cor(train$Num_Social_Media_Platforms, train$Visa_Approved))

############################################################
## 5. Baseline models: glm & rpart
##    -> “student-level” models without feature engineering
############################################################

## Convert Visa_Approved to factor for rpart / RF
train$Visa_Approved <- factor(train$Visa_Approved, levels = c(0, 1))
test$Visa_Approved  <- factor(test$Visa_Approved,  levels = c(0, 1))

## 5.1. Logistic regression (glm) with main effects only
glm_fit <- glm(Visa_Approved ~ Age + Gender + Region + Family_Income +
                 Program_Level + Field_of_Study + GPA + IELTS_Score +
                 Num_Countries_Visited + Funding_Type + Has_sibling +
                 Has_US_Relative + Season_Applied +
                 Num_Social_Media_Platforms,
               data = train, family = binomial)

glm_prob <- predict(glm_fit, newdata = test, type = "response")
glm_pred_num <- ifelse(glm_prob > 0.5, 1, 0)
glm_pred <- factor(glm_pred_num, levels = c(0, 1))

glm_acc <- mean(glm_pred == test$Visa_Approved)
cat("\nGLM test accuracy:\n")
print(glm_acc)

## 5.2. Decision tree (rpart) – simple non-linear baseline
rp_fit <- rpart(Visa_Approved ~ Age + Gender + Region + Family_Income +
                  Program_Level + Field_of_Study + GPA + IELTS_Score +
                  Num_Countries_Visited + Funding_Type + Has_sibling +
                  Has_US_Relative + Season_Applied +
                  Num_Social_Media_Platforms,
                data = train, method = "class",
                control = rpart.control(cp = 0.01, minsplit = 30))

rp_pred <- predict(rp_fit, newdata = test, type = "class")
rp_acc <- mean(rp_pred == test$Visa_Approved)
cat("\nrpart test accuracy:\n")
print(rp_acc)

############################################################
## 6. Strong baseline: Random Forest
##    -> better non-linear model with default features
############################################################

set.seed(2025)
rf_fit <- randomForest(
  Visa_Approved ~ Age + Gender + Region + Family_Income +
    Program_Level + Field_of_Study + GPA + IELTS_Score +
    Num_Countries_Visited + Funding_Type + Has_sibling +
    Has_US_Relative + Season_Applied + Num_Social_Media_Platforms,
  data = train,
  ntree = 600,   # number of trees
  mtry  = 4,     # number of variables tried at each split
  nodesize = 8   # minimum terminal node size
)

rf_pred <- predict(rf_fit, newdata = test)
rf_acc <- mean(rf_pred == test$Visa_Approved)
cat("\nRandom Forest test accuracy:\n")
print(rf_acc)

############################################################
## 7. Solution model: engineered features (~95% expected)
##    -> here we “reverse engineer” the hidden rules
############################################################

## Create new features on numeric copies train_num, test_num

## 7.1 Academic_Index – GPA + IELTS combined again
train_num$Academic_Index <- 0.6 * (train_num$GPA - 3.0) +
  0.4 * (train_num$IELTS_Score - 6.5)
test_num$Academic_Index  <- 0.6 * (test_num$GPA - 3.0) +
  0.4 * (test_num$IELTS_Score - 6.5)

## Quadratic term to capture non-linearity
train_num$Academic_Index2 <- train_num$Academic_Index^2
test_num$Academic_Index2  <- test_num$Academic_Index^2

## 7.2 Log_Income – same transformation used in the rules
train_num$Log_Income <- log10(train_num$Family_Income + 1)
test_num$Log_Income  <- log10(test_num$Family_Income + 1)

## 7.3 Mobility_Index – similar idea to mobility_raw
train_num$Mobility_Index <- 0.1 * train_num$Num_Countries_Visited +
  1.0 * (train_num$Has_US_Relative == "Yes")
test_num$Mobility_Index  <- 0.1 * test_num$Num_Countries_Visited +
  1.0 * (test_num$Has_US_Relative == "Yes")

## 7.4 Funding_Score (ordinal encoding of funding strength)
train_num$Funding_Score <- ifelse(train_num$Funding_Type == "Scholarship", 3,
                                  ifelse(train_num$Funding_Type == "Mixed", 2,
                                         ifelse(train_num$Funding_Type == "Family", 1, 0)))
test_num$Funding_Score  <- ifelse(test_num$Funding_Type == "Scholarship", 3,
                                  ifelse(test_num$Funding_Type == "Mixed", 2,
                                         ifelse(test_num$Funding_Type == "Family", 1, 0)))

## 7.5 Season_Score – encode Fall > Spring > Summer
train_num$Season_Score <- ifelse(train_num$Season_Applied == "Fall", 2,
                                 ifelse(train_num$Season_Applied == "Spring", 1, 0))
test_num$Season_Score  <- ifelse(test_num$Season_Applied == "Fall", 2,
                                 ifelse(test_num$Season_Applied == "Spring", 1, 0))

## 7.6 Age_Mismatch indicator – copy of rule logic
train_num$Age_Mismatch <- as.integer(
  (train_num$Program_Level == "Undergrad" & train_num$Age > 28) |
    (train_num$Program_Level == "Masters"   & train_num$Age < 23) |
    (train_num$Program_Level == "PhD"       & train_num$Age < 25)
)
test_num$Age_Mismatch <- as.integer(
  (test_num$Program_Level == "Undergrad" & test_num$Age > 28) |
    (test_num$Program_Level == "Masters"   & test_num$Age < 23) |
    (test_num$Program_Level == "PhD"       & test_num$Age < 25)
)

## Convert Visa_Approved to factor for glm solution
train_num$Visa_Approved <- factor(train_num$Visa_Approved, levels = c(0, 1))
test_num$Visa_Approved  <- factor(test_num$Visa_Approved,  levels = c(0, 1))

## 7.7 GLM solution with interaction Program_Level * engineered features
## Idea: different program levels use academic / income / mobility differently.
'glm_solution <- glm(
  Visa_Approved ~ Program_Level * (Academic_Index + Academic_Index2) +
    Log_Income +
    Mobility_Index +
    Funding_Score +
    Season_Score +
    Age_Mismatch,
  data = train_num,
  family = binomial
)'

glm_solution <- glm(
  Visa_Approved ~ Program_Level * (Academic_Index + Academic_Index2 +
                                     Log_Income + Mobility_Index +
                                     Funding_Score + Season_Score) +
    Age_Mismatch,
  data = train_num,
  family = binomial
)

sol_prob <- predict(glm_solution, newdata = test_num, type = "response")
sol_pred_num <- ifelse(sol_prob > 0.5, 1, 0)
sol_pred <- factor(sol_pred_num, levels = c(0, 1))

sol_acc <- mean(sol_pred == test_num$Visa_Approved)
cat("\nEngineered GLM solution test accuracy (target ~0.95):\n")
print(sol_acc)

############################################################
## 8. Oracle / upper bound
##    -> re-apply the exact generating rules as a classifier
############################################################

compute_score <- function(df_local) {
  ## Recompute the same internal pieces used earlier
  academic_raw <- 0.6 * (df_local$GPA - 3.0) +
    0.4 * ((df_local$IELTS_Score - 6.5) / 1.5)
  
  income_log      <- log10(df_local$Family_Income + 1)
  income_centered <- income_log - 4.7
  
  mobility_raw <- 0.10 * df_local$Num_Countries_Visited +
    0.60 * (df_local$Has_US_Relative == "Yes")
  
  funding_bonus <- ifelse(df_local$Funding_Type == "Scholarship", 1.0,
                          ifelse(df_local$Funding_Type == "Mixed",      0.4,
                                 ifelse(df_local$Funding_Type == "Family",     0.3, 0.0)))
  
  season_effect <- ifelse(df_local$Season_Applied == "Fall",   0.30,
                          ifelse(df_local$Season_Applied == "Spring", 0.00, -0.30))
  
  age_penalty <- ifelse(df_local$Program_Level == "Undergrad" & df_local$Age > 28, -0.25,
                        ifelse(df_local$Program_Level == "Masters"   & df_local$Age < 23, -0.25,
                               ifelse(df_local$Program_Level == "PhD"       & df_local$Age < 25, -0.25, 0.0)))
  
  score_local <- numeric(nrow(df_local))
  
  idx_u <- df_local$Program_Level == "Undergrad"
  idx_m <- df_local$Program_Level == "Masters"
  idx_p <- df_local$Program_Level == "PhD"
  
  score_local[idx_u] <-
    -0.20 +
    1.20 * academic_raw[idx_u] +
    0.50 * funding_bonus[idx_u] +
    0.20 * mobility_raw[idx_u] -
    0.30 * (df_local$GPA[idx_u] < 2.8)
  
  score_local[idx_m] <-
    -0.50 +
    1.40 * academic_raw[idx_m] +
    0.30 * income_centered[idx_m] +
    0.25 * mobility_raw[idx_m] +
    0.30 * (df_local$Field_of_Study[idx_m] %in% c("STEM", "Business"))
  
  score_local[idx_p] <-
    -0.60 +
    1.70 * academic_raw[idx_p] +
    0.35 * income_centered[idx_p] +
    0.30 * mobility_raw[idx_p] +
    0.70 * (df_local$Field_of_Study[idx_p] == "STEM") +
    0.35 * (df_local$Funding_Type[idx_p] == "Scholarship") -
    0.20 * (df_local$Has_sibling[idx_p] >= 3)
  
  score_local <- score_local + season_effect + age_penalty
  score_local
}

solution_predict <- function(df_local) {
  ## Convert score back to 0/1 classification
  score_local <- compute_score(df_local)
  ifelse(score_local > 0, 1L, 0L)
}

## Evaluate oracle on test_num (0/1 numeric label)
test_oracle <- test_num
test_oracle$Visa_Approved <- as.integer(as.character(test_num$Visa_Approved))

oracle_pred <- solution_predict(test_oracle)
oracle_acc  <- mean(oracle_pred == test_oracle$Visa_Approved)
cat("\nOracle (rule-based) test accuracy (upper bound ~0.98 expected):\n")
print(oracle_acc)

