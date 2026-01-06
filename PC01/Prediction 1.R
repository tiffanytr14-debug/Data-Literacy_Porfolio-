install.packages("rpart")
install.packages("rpart.plot")
install.packages("ModelMetrics")
install.packages("devtools")
install.packages("bestNormalize")
install.packages("randomForest") 
library(randomForest)
library(CrossValidation)

setwd("~/Documents/Data101")

library(rpart)
library(rpart.plot)
library(dplyr)
library(CrossValidation)
library(bestNormalize)

train <- read.csv("CarsTrainNew.csv")
test <- read.csv("CarsTestNew+Truncated.csv")

clean_data <- function(df) {
  df <- df %>%
    mutate(across(where(is.character) & !matches("Deal"),
                  ~ tolower(trimws(.)))) %>%
    mutate(across(!where(is.character), as.numeric))
  return(df)
}


train <- clean_data(train)
test  <- clean_data(test)
train$Deal <- factor(train$Deal)

train$Deal

summary(train)

train |> summarise(across(everything(), ~ sum(is.na(.x))))


str(train)
colnames(train)
colSums(is.na(test))



train$log_price <- log(pmax(train$Price, 1))

train$log_mileage <- log(pmax(train$Mileage, 0) + 1)

train$ratio_pm <- train$log_price - train$log_mileage

# (1)  lệch log  Price and Benchmark
train$DiffLogPriceBench <- train$log_price - train$ValueBenchmark


# (2) abs value
train$AbsDiffLogPriceBench <- abs(train$DiffLogPriceBench)

# (6) Interaction benchmark and log_mileage
train$VB_x_Mileage <- train$ValueBenchmark * train$Mileage
#(7) Overprice
train$OverPrice_xM <- train$AbsDiffLogPriceBench * train$Mileage


train$LogP_xM <- train$log_price * train$Mileage



unique(train$Model)
unique(train$Location)

table(train$Make)
table(test$Make)


table(train$Model)
table(test$Model)


table(train$Location)
table(test$Location)


# Top 10 
head(train[order(-train$Price), c("Make", "Model", "Price", "Mileage", "Location","Deal")], 10)

aggregate(Price ~ Make, data = train, mean, na.rm = TRUE) |>
  transform(Price = round(Price, 0)) 


#Gprice decending
aggregate(Price ~ Make, data = train, mean, na.rm = TRUE)[order(-aggregate(Price ~ Make, data = train, mean, na.rm = TRUE)$Price), ]

#price base on location
aggregate(Price ~ Location, data = train, mean, na.rm = TRUE)[order(-aggregate(Price ~ Location, data = train, mean, na.rm = TRUE)$Price), ]

table(train$Make, train$Deal)


#scatter
plot(train$Mileage, train$Price,
     main = "Price vs Mileage",
     xlab = "Mileage",
     ylab = "Price",
     col = "blue",
     pch = 19)


plot(train$ValueBenchmark, train$Price,
     main = "Price vs ValueBenchmark",
     xlab = "ValueBenchmark",
     ylab = "Price",
     col = "darkgreen",
     pch = 19)


plot(train$ValueBenchmark, train$Mileage,
     main = "Mileage vs ValueBenchmark",
     xlab = "ValueBenchmark",
     ylab = "Mileage",
     col = "orange",
     pch = 19)


plot(train$ValueBenchmark, train$log_mileage,
     main = "Log Mileage vs  ValueBenchmark",
     xlab = "ValueBenchmark",
     ylab = "Mileage",
     col = "orange",
     pch = 19)

#boxplot
boxplot(Price ~ Deal, data = train,
        main = "Price theo Deal",
        xlab = "Deal",
        ylab = "Price",
        col = "lightblue")

boxplot(Mileage ~ Deal, data = train,
        main = "Mileage theo Deal",
        xlab = "Deal",
        ylab = "Mileage",
        col = "lightgreen")
#log 
boxplot(log_mileage ~ Deal, data = train,
        main = "Mileage theo Deal",
        xlab = "Deal",
        ylab = "Mileage",
        col = "lightgreen")


boxplot(ValueBenchmark ~ Deal, data = train,
        main = "ValueBenchmark theo Deal",
        xlab = "Deal",
        ylab = "ValueBenchmark",
        col = "lightpink")


#Rpart ->
cv_df <- train[, c("Deal", "Price", "Mileage", "ValueBenchmark", "ratio_pm", "log_price", "AbsDiffLogPriceBench")]
head(cv_df)
tree_deal <- rpart(
  Deal ~ log_price + Mileage + ValueBenchmark + ratio_pm,
  data   = cv_df,
  method = "class",
  control = rpart.control(
    xval = 0,
    maxdepth = 30,
    minsplit = 7
  )
)
rpart.plot(tree_deal, main = "Decision tree: Deal ~ Features")

train_pred_class <- predict(tree_deal, newdata = cv_df, type = "class")  
table(train$Deal, train_pred_class)
mean(train$Deal == train_pred_class, na.rm=TRUE)


cv_output <-CrossValidation::cross_validate(cv_df, tree_deal, 10, 0.8)


cv_output[[1]]
cv_output[[2]]



#Randomforest
rf_df <- train[, c("Deal","Mileage","AbsDiffLogPriceBench", "LogP_xM", "VB_x_Mileage","OverPrice_xM"  )]

#Random Forest  + xem OOB & train accuracy
set.seed(101)

p <- ncol(rf_df) - 1   # số feature (trừ Deal)
mtry_default <- floor(sqrt(p))

rf_basic <- randomForest(
  Deal ~ .,
  data      = rf_df,
  ntree     = 500,        # number of tree
  mtry      = mtry_default,
  nodesize  = 5,          # min  (same as minbucket 
  importance = TRUE
)

print(rf_basic)   # OOB error & confusion matrix

# OOB accuracy (thực tế hơn train-acc)
oob_err  <- rf_basic$err.rate[rf_basic$ntree, "OOB"]
oob_acc  <- 1 - oob_err
oob_acc


# Train accuracy ( > 0.9)
rf_train_pred <- predict(rf_basic, newdata = rf_df, type = "class")
rf_train_acc  <- mean(rf_train_pred == rf_df$Deal)
rf_train_acc




set.seed(101)

mtry_grid    <- 2:5        
nodesize_grid <- c(1, 3, 5, 10)

result_list <- data.frame(
  mtry     = integer(),
  nodesize = integer(),
  oob_acc  = numeric(),
  train_acc = numeric()
)

for (m in mtry_grid) {
  for (ns in nodesize_grid) {
    cat("Training RF with mtry =", m, "nodesize =", ns, "...\n")
    
    rf_tmp <- randomForest(
      Deal ~ .,
      data      = rf_df,
      ntree     = 400,   
      mtry      = m,
      nodesize  = ns
    )
    
    # OOB accuracy
    oob_err_tmp <- rf_tmp$err.rate[rf_tmp$ntree, "OOB"]
    oob_acc_tmp <- 1 - oob_err_tmp
    
    # Train accuracy
    pred_tmp    <- predict(rf_tmp, rf_df, type = "class")
    train_acc_tmp <- mean(pred_tmp == rf_df$Deal)
    
    result_list <- rbind(
      result_list,
      data.frame(
        mtry     = m,
        nodesize = ns,
        oob_acc  = oob_acc_tmp,
        train_acc = train_acc_tmp
      )
    )
  }
}

result_list[order(-result_list$oob_acc), ]


#Train final model 


best_mtry    <- 2   
best_nodesize <- 10 

set.seed(101)
rf_final <- randomForest(
  Deal ~ .,
  data      = rf_df,
  ntree     = 800,          
  mtry      = best_mtry,
  nodesize  = best_nodesize,
  importance = TRUE
)

print(rf_final)
oob_err_final <- rf_final$err.rate[rf_final$ntree, "OOB"]
oob_acc_final <- 1 - oob_err_final
oob_acc_final

rf_final_pred_train <- predict(rf_final, rf_df, type = "class")
rf_final_train_acc  <- mean(rf_final_pred_train == rf_df$Deal)
rf_final_train_acc


importance(rf_final)
varImpPlot(rf_final, main = "Random Forest - Variable Importance")


#CV
set.seed(101)
K <- 10
n <- nrow(rf_df)


folds <- sample(rep(1:K, length.out = n))

cv_acc <- numeric(K)

for (k in 1:K) {
  cat("Fold", k, "...\n")
  
  train_idx <- which(folds != k)
  test_idx  <- which(folds == k)
  
  rf_k <- randomForest(
    Deal ~ .,
    data      = rf_df[train_idx, ],
    ntree     = 800,
    mtry      = best_mtry,     # 5
    nodesize  = best_nodesize  # 5
  )
  
  pred_k <- predict(rf_k, newdata = rf_df[test_idx, ], type = "class")
  cv_acc[k] <- mean(pred_k == rf_df$Deal[test_idx])
}

cv_acc         
mean(cv_acc)   
var(cv_acc)    



#submit
test$log_price    <- log(pmax(test$Price, 1))
test$log_mileage  <- log(pmax(test$Mileage, 0) + 1)

test$ratio_pm <- test$log_price - test$log_mileage

test$DiffLogPriceBench    <- test$log_price - test$ValueBenchmark
test$AbsDiffLogPriceBench <- abs(test$DiffLogPriceBench)

test$VB_x_Mileage <- test$ValueBenchmark * test$Mileage
test$OverPrice_xM <- test$AbsDiffLogPriceBench * test$Mileage

test$LogP_xM <- test$log_price * test$Mileage




rf_df <- train[, c("Deal","Mileage","AbsDiffLogPriceBench", "LogP_xM", "VB_x_Mileage","OverPrice_xM")]


test_rf_df <- test[, c("Mileage", "AbsDiffLogPriceBench", 
                       "LogP_xM", "VB_x_Mileage", "OverPrice_xM")]
test_pred <- predict(rf_final, 
                     newdata = test_rf_df, 
                     type = "class")
test_pred

sum(is.na(test_pred))

colnames(test)


#Generate submission file
submission <- data.frame(
  ID   = test$id,               
  Deal = as.character(test_pred)
)

write.csv(submission,
          file = "submission.csv",
          row.names = FALSE)


test_original <- read.csv("CarsTestNew+Truncated.csv")
test_original <- clean_data(test_original)

# add a deal col
test_original$Deal <- as.character(test_pred)


write.csv(
  test_original,
  file = "test_with_predictions.csv",
  row.names = FALSE
)


