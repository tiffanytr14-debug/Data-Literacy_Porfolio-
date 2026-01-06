AcademicStress <- read.csv("~/Documents/Data101/academic Stress level - maintainance 1.csv")
head(AcademicStress)
nrow(AcademicStress)
names(AcademicStress)

nrow(AcademicStress)

head(AcademicStress)

colnames(AcademicStress)

#remove the timeframe col 
AcademicStress <- subset(AcademicStress, select = -Timestamp)

ncol(AcademicStress)
colnames(AcademicStress)
#basic functions on the numeric outcome 
min(AcademicStress$StressIndex)
max(AcademicStress$StressIndex)
mean(AcademicStress$StressIndex)

unique(AcademicStress$Environment)

#rename col
names(AcademicStress) <- c("Stage","PeerPressure","HomePressure","Environment","CopingStrategy","BadHabits","Competition","StressIndex")

#found a missing value from data set 
AcademicStress$Environment[130] <- "Unknown"
#rename(capitalize) to align the categories
AcademicStress$Environment[AcademicStress$Environment=="disrupted"] <- "Disrupted"

#group means
tapply(AcademicStress$StressIndex, AcademicStress$Environment, mean)
tapply(AcademicStress$StressIndex, AcademicStress$BadHabits, mean)
tapply(AcademicStress$StressIndex, AcademicStress$CopingStrategy, mean)
tapply(AcademicStress$StressIndex, AcademicStress$PeerPressure,mean)
#counts
#table for checking counts
table(AcademicStress$Environment)
table(AcademicStress$BadHabits)
table(AcademicStress$Stage)

#1st plot boxplot for each environment and stress index
boxplot(StressIndex~ Environment, data= AcademicStress, main = "Stress vs Study Environment",
        xlab="Enivronment",
        ylab="Stress Index",
        col=c("lightcoral","lightsalmon","lightblue","lightgray"))

env_means <- tapply(AcademicStress$StressIndex, AcademicStress$Environment,mean)

barplot(env_means, main="Average Stress by Environment",
        xlab="Environment",
        ylab="Mean Stress",
        col=c("lightcoral","lightsalmon","lightblue","lightgray"))

#scatter plot 
plot(AcademicStress$PeerPressure, AcademicStress$StressIndex,
     main="Peer Pressure vs Stress", 
     xlab="Peer Pressure (1–5)",
     ylab="Stress Index (1–5)", 
     col=c("red","orange","lightblue","lightgray"))
#it looks weird cause only 25 possible paries. cannot see how many students piled on each dot. doesnot make sense

boxplot(StressIndex ~ PeerPressure, data=AcademicStress, 
        main="Stress by Peer Pressure",
        xlab="Peer Pressure(1-5)",
        ylab="Stress Index(1-5",
        col=c("lightgreen","lightblue","lightsalmon","lightcoral","red"))

boxplot(StressIndex ~ BadHabits, data=AcademicStress,
        main="Stress by Daily Habit (Distribution)",
        xlab="Daily Smoking/Drinking?", 
        ylab="Stress Index (1–5)",
        col=c("lightblue","lightgray","lightcoral"))
