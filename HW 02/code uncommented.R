install.packages("devtools")
library(devtools)
devtools::install_github("janish-parikh/ZTest")
library(HypothesisTesting)


students <- read.csv("~/Downloads/student_habits_performance.csv")
df <- subset( students, (internet_quality == "Average" | internet_quality == "Good") & !is.na(exam_score)
)
df$internet_quality <- factor(df$internet_quality, levels = c("Good","Average"))


m_average  <- mean(df$exam_score[df$internet_quality == "Average"])
m_good <- mean(df$exam_score[df$internet_quality == "Good"])
obs_diff <- m_average - m_good
obs_diff  

p_perm <- permutation_test(df, 'internet_quality', 'exam_score', 10000, 'Good', 'Average')

p_z <- z_test_from_data(df, 'internet_quality', 'exam_score', 'Good', 'Average')

cat(sprintf("\nObserved difference (Average - Poor): %.3f\n", obs_diff))
cat(sprintf("Permutation test p-value: %s\n", as.character(p_perm)))
cat(sprintf("Z-test p-value: %s\n", as.character(p_z)))


#Confidence Interval 
z_score = 1.96
attendance_std <- subset(students, part_time_job == "No" & extracurricular_participation == "No"  & !is.na(attendance_percentage) & age >=18 & age <=22 ) 
n  <- nrow(attendance_std)
m  <- mean(attendance_std$attendance_percentage)
s  <- sd(attendance_std$attendance_percentage)
sem <- s / (n^(1/2))

moe <- z_score * sem

round(c(n = n, mean = m, sd = s, SEM = sem, MOE = moe, LCL = m - moe, UCL = m + moe), 2)

# Chi-Square Test

cont1 <- table (students$gender,students$internet_quality)
cont1

chisq1 <- chisq.test(cont1)
chisq1

chisq1$expected
chisq1$observed

chi_sq <- unname(chisq1$statistic)   
df     <- unname(chisq1$parameter)  
pval   <- unname(chisq1$p.value)    

cat(sprintf("Chi-square (χ²) = %.3f, df = %d, p-value = %.4f\n", chi_sq, df, pval))

#BayseanReasoning
df <- subset(students, !is.na(exam_score) & !is.na(attendance_percentage))

HighScore <- df$exam_score >= 80
LowSocial <- df$social_media_hours < 0.5  

# Prior
PriorP <- mean(HighScore)
PriorOdds <- PriorP / (1 - PriorP)

# Likelihood Ratio
TP <- sum(LowSocial & HighScore) / sum(HighScore)
FP <- sum(LowSocial & !HighScore) / sum(!HighScore)
LR <- TP / FP

# Posterior
PostOdds <- LR * PriorOdds
PostP <- PostOdds / (1 + PostOdds)

# Results
cat("=== Bayesian Reasoning: Social Media < 0.5h/day ===\n")
cat(sprintf("Prior P(HighScore): %.3f (%.1f%%)\n", PriorP, PriorP*100))
cat(sprintf("Prior Odds: %.3f\n", PriorOdds))
cat(sprintf("Likelihood Ratio: %.3f\n", LR))
cat(sprintf("Posterior Odds: %.3f\n", PostOdds))
cat(sprintf("Posterior P(HighScore|LowSocial): %.3f (%.1f%%)\n\n", PostP, PostP*100))

