setwd("~/Documents/Data101")

getwd()

install.packages("rpart")
install.packages("rpart.plot")
install.packages("ranger")
install.packages("caret")
install.packages("ggplot2")

library(rpart)
library(rpart.plot)
library(ranger)
library(caret)

train <- read.csv("~/Documents/Data101/earnings_train copy.csv")
test  <- read.csv("~/Documents/Data101/earnings_test.csv")

if (!"Earnings" %in% colnames(train)) {
  stop("Target column 'Earnings' not found in training data.")
}

num_cols <- c("GPA", "Height", "Number_Of_Professional_Connections",
              "Graduation_Year", "Number_Of_Credits", "Number_Of_Parking_Tickets")

for (col in num_cols) {
  if (col %in% colnames(train)) {
    train[[col]] <- as.numeric(as.character(train[[col]]))
  }
  if (col %in% colnames(test)) {
    test[[col]] <- as.numeric(as.character(test[[col]]))
  }
}

cat_cols <- c("Major")

for (col in cat_cols) {
  if (col %in% colnames(train)) {
    train[[col]] <- as.factor(train[[col]])
  }
  if (col %in% colnames(test)) {
    test[[col]] <- as.factor(test[[col]])
  }
}
train$GPA <- log1p(train$GPA)
test$GPA  <- log1p(test$GPA)

predictor_names <- setdiff(names(train), "Earnings")

nzv_metrics <- caret::nearZeroVar(train[, predictor_names, drop = FALSE],
                                  saveMetrics = TRUE)

if (any(nzv_metrics$nzv)) {
  remove_cols <- rownames(nzv_metrics)[nzv_metrics$nzv]
  message("Removing near-zero-variance columns: ",
          paste(remove_cols, collapse = ", "))
  
  train <- train[, !(names(train) %in% remove_cols)]
  test  <- test[,  !(names(test)  %in% remove_cols)]
}

print("Training ranger regression model (fixed hyperparameters)...")

rf_model <- ranger(
  formula        = Earnings ~ .,
  data           = train,
  num.trees      = 500,
  mtry           = 3,
  min.node.size  = 10,
  splitrule      = "variance",
  importance     = "impurity"
)

print(rf_model)

train_model_df <- train

train_pred <- predict(rf_model, data = train_model_df)$predictions

MSE_train <- mean((train_model_df$Earnings - train_pred)^2)
MSE_train
cat("MSE on training set:", MSE_train, "\n")
test_pred <- predict(rf_model, data = test)$predictions

test$ID <- 1:nrow(test)
test$Earnings <- test_pred

set.seed(123)
K <- 5
n <- nrow(train)

fold_id <- sample(rep(1:K, length.out = n))
cv_mse <- numeric(K)

for (k in 1:K) {
  train_idx <- which(fold_id != k)
  test_idx  <- which(fold_id == k)
  
  train_fold <- train[train_idx, ]
  test_fold  <- train[test_idx, ]
  
  rf_k <- ranger(
    formula        = Earnings ~ .,
    data           = train_fold,
    num.trees      = 500,
    mtry           = 3,
    min.node.size  = 10,
    splitrule      = "variance"
  )
  
  pred_k <- predict(rf_k, data = test_fold)$predictions
  
  cv_mse[k] <- mean((test_fold$Earnings - pred_k)^2)
}

cat("MSE cho từng fold:\n")

print(cv_mse)
mean_cv_mse <- mean(cv_mse)
cat("Mean CV MSE (5-fold):", mean_cv_mse, "\n")

test_pred <- predict(rf_model, data = test)$predictions

if (file.exists("cv_model9.rds")) {
  print("Loading cross-validated model from file...")
  cv_model <- readRDS("cv_model9.rds")
} else {
  print("Performing cross-validation...")
  
  cv_folds <- 5
  
  set.seed(42)
  
  customSummary <- function(data, lev = NULL, model = NULL) {
    if (!all(c("obs", "pred") %in% names(data))) {
      return(c(MSE = NA, RMSE = NA, Rsquared = NA))
    }
    mse  <- mean((data$obs - data$pred)^2, na.rm = TRUE)
    rmse <- sqrt(mse)
    rsq  <- tryCatch({
      if (length(unique(data$obs)) > 1)
        cor(data$obs, data$pred, use = "complete.obs")^2
      else NA
    }, error = function(e) NA)
    c(MSE = mse, RMSE = rmse, Rsquared = as.numeric(rsq))
  }
  
  cv_control <- trainControl(
    method          = "cv",
    number          = cv_folds,
    summaryFunction = customSummary,
    savePredictions = "final",
    allowParallel   = TRUE
  )
  
  p <- ncol(train) - 1
  
  mtry_vals <- 3
  
  tgrid <- expand.grid(
    mtry          = mtry_vals,
    splitrule     = "variance",
    min.node.size = 10
  )
  
  cv_model <- train(
    Earnings ~ .,
    data      = (function(d) { if ("Predicted" %in% names(d)) d$Predicted <- NULL; d })(train),
    method    = "ranger",
    trControl = cv_control,
    tuneGrid  = tgrid,
    metric    = "MSE",
    num.trees = 500
  )
  
  saveRDS(cv_model, "cv_model9.rds")
}
print(cv_model)

print("Plotting cross-validation results...")

test$ID <- 1:nrow(test)
test$Earnings <- test_pred

write.csv(
  test[, c("ID", "Earnings")],
  "submission.csv",
  row.names = FALSE
)


