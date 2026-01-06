install.packages("devtools")
library(devtools)
devtools::install_github("janish-parikh/ZTest")
library(HypothesisTesting)

jobs <-  read.csv("~/Documents/AI_Impact_on_Jobs_2030.csv", stringsAsFactors = TRUE)

# Check structure
str(jobs)
names(jobs)
summary(jobs)

# 1) Basic summaries -----------------------------------------

# How many rows and columns?
dim(jobs)

# Categorical distributions
table(jobs$Risk_Category)

table(jobs$Education_Level)

# Numeric summaries
summary(jobs$Average_Salary)
summary(jobs$AI_Exposure_Index)

# Example subset: only High-Risk jobs
high_risk_jobs <- subset(jobs, Risk_Category == "High")
nrow(high_risk_jobs)

# tapply: mean salary by risk category
tapply(jobs$Average_Salary, jobs$Risk_Category, mean)

jobs$Automation_Risk_Score <- jobs$AI_Exposure_Index *
  jobs$Automation_Probability_2030 * 100

# 2) Plots ---------------------------------------------------

# Histogram of salaries
hist(jobs$Average_Salary,
     main = "Distribution of Average Salary",
     xlab = "Average Salary")

# Barplot of risk categories
risk_counts <- table(jobs$Risk_Category)
barplot(risk_counts,
        main = "Number of Jobs by Automation Risk Category",
        ylab = "Count")

#  boxplot: salary by risk category
boxplot(Average_Salary ~ Risk_Category,
        data = jobs,
        main = "Salaries by Automation Risk Category",
        xlab = "Risk Category",
        ylab = "Average Salary")
# 3) Fooled by randomness ------------------------------------

# Real correlation
cor_real <- cor(jobs$AI_Exposure_Index,
                jobs$Automation_Probability_2030)
cor_real

# Shuffle AI_Exposure_Index randomly
set.seed(123)
jobs$AI_Exposure_Shuffled <- sample(jobs$AI_Exposure_Index)

# Correlation after shuffling (should be near 0)
cor_shuffled <- cor(jobs$AI_Exposure_Shuffled,
                    jobs$Automation_Probability_2030)
cor_shuffled

#  tiny scatterplots 
plot(jobs$AI_Exposure_Index,
     jobs$Automation_Probability_2030,
     main = "Real Data: AI Exposure vs Automation Probability",
     xlab = "AI Exposure Index",
     ylab = "Automation Probability 2030")


plot(jobs$AI_Exposure_Shuffled,
     jobs$Automation_Probability_2030,
     main = "Shuffled AI Exposure (Random Noise)",
     xlab = "Shuffled AI Exposure",
     ylab = "Automation Probability 2030")


# Part 4 – Central Limit Theorem & Confidence Interval


# 1) Look at the population of interest: Average_Salary ----

summary(jobs$Average_Salary)
sd(jobs$Average_Salary)

# 2) Central Limit Theorem: sampling many times ----

set.seed(123)      # so results are reproducible

n  <- 50           # sample size (>= 30)
B  <- 200          # number of repeated samples
sample_means <- numeric(B)  # empty vector to store means

for (i in 1:B) {
  # sample row indices with replacement
  rows <- sample(1:nrow(jobs), size = n, replace = TRUE)
  
  # compute the mean salary in this sample
  sample_means[i] <- mean(jobs$Average_Salary[rows])
}

# Look at the sampling distribution of the mean
hist(sample_means,
     main = "Sampling Distribution of Mean Salary (n = 50)",
     xlab  = "Sample Mean Salary")

mean(sample_means)   # mean of sample means
sd(sample_means)     # sd of sample means (should be close to SD/sqrt(n))


# 3) 95% Confidence Interval for the mean salary ----

x      <- jobs$Average_Salary
n      <- length(x)
mean_x <- mean(x)
sd_x   <- sd(x)

SE  <- sd_x / sqrt(n)      # standard error of the mean
z   <- 1.96                # 95% confidence
lower <- mean_x - z * SE
upper <- mean_x + z * SE

mean_x
lower
upper

# 5) Hypothesis test: permutation ----------------------------
# Keep only Low and High risk, and non-missing automation probability
# Keep only Low and High risk + non-missing automation prob
# Keep only Bachelor's and Master's with non-missing salary
unique(jobs$Education_Level)

# Keep only Bachelor's and Master's with non-missing salary
df <- subset(
  jobs,
  (Education_Level == "Bachelor's" | Education_Level == "Master's") &
    !is.na(Average_Salary)
)

# Order factor levels: Bachelor's, then Master's
df$Education_Level <- factor(df$Education_Level,
                             levels = c("Bachelor's", "Master's"))

# Group means
m_ba <- mean(df$Average_Salary[df$Education_Level == "Bachelor's"])
m_ma <- mean(df$Average_Salary[df$Education_Level == "Master's"])

# Observed difference (Master's - Bachelor's)
obs_diff <- m_ma - m_ba
obs_diff   # should be about 2500, not NaN

# Permutation test (POSitional arguments, no names)
p_perm <- permutation_test(
  df,
  "Education_Level",   # group column
  "Average_Salary",    # value column
  10000,               # number of permutations
  "Bachelor's",        # group1
  "Master's"           # group2
)

# z-style test
p_z <- z_test_from_data(
  df,
  "Education_Level",   # group column
  "Average_Salary",    # value column
  "Bachelor's",        # group1
  "Master's"           # group2
)

cat(sprintf("\nObserved difference (Master's - Bachelor's): %.2f\n", obs_diff))
cat(sprintf("Permutation test p-value: %s\n", as.character(p_perm)))
cat(sprintf("Z-test p-value: %s\n", as.character(p_z)))

# 5. Chi-Square Test: Education Level vs Risk Category -----------------------

# Keep only rows where both variables are not missing
df_chi <- subset(jobs,
                 !is.na(Education_Level) & !is.na(Risk_Category))

# 1. Make the contingency table
cont1 <- table(df_chi$Education_Level, df_chi$Risk_Category)
cont1   # look at counts

# 2. Run chi-square test of independence
chisq1 <- chisq.test(cont1)
chisq1  # full output

# 3. Look at expected vs observed counts (optional)
chisq1$expected
chisq1$observed

# 4. Pull out statistic, df, and p-value nicely
chi_sq <- unname(chisq1$statistic)
df     <- unname(chisq1$parameter)
pval   <- unname(chisq1$p.value)

cat(sprintf("Chi-square (χ²) = %.3f, df = %d, p-value = %.4f\n",
            chi_sq, df, pval))
