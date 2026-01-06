############################################################
## 0. Setup: read data AND sort by ID first
############################################################

train <- read.csv("~/Desktop/pred3data101/PredictionTrain.csv")
test  <- read.csv("~/Desktop/pred3data101/PredictionTest.csv")
loc   <- read.csv("~/Desktop/pred3data101/Location.csv")

## Sort train and test by ID BEFORE anything else
train <- train[order(train$ID), ]
test  <- test[order(test$ID), ]

## Merge State into train via University
train <- merge(train,
               loc[, c("University", "State")],
               by = "University",
               all.x = TRUE)

str(train)

############################################################
## ---------- 1. FEATURE ENGINEERING HELPERS ----------
############################################################

## Humanities / social science majors
hum_majors <- c("Literature",
                "Philosophy",
                "Political Science",
                "Sociology")

## Choose which states you treat as "blue".
blue_states <- c("CA","CO","CT","DE","HI","IL","ME","MD","MA",
                 "MN","NJ","NM","NY","OR","RI","VA","VT","WA")

add_features <- function(df,
                         hum_majors,
                         blue_states,
                         gpa_cut = 2.85) {
  df$Humanities <- ifelse(df$Major %in% hum_majors, 1L, 0L)
  df$GPA_high   <- ifelse(df$GPA >= gpa_cut, 1L, 0L)
  df$BlueState  <- ifelse(df$State %in% blue_states, 1L, 0L)
  df
}

train_feat <- add_features(train, hum_majors, blue_states, gpa_cut = 2.85)

## Make sure Hired is numeric 0/1
if (is.factor(train_feat$Hired)) {
  train_feat$Hired <- as.integer(as.character(train_feat$Hired))
}

############################################################
## ---------- 2. SIMPLE RULE MODEL ----------
############################################################

rule_predict_simple <- function(df, gpa_cut = 2.85,
                                hum_majors, blue_states) {
  df <- add_features(df, hum_majors, blue_states, gpa_cut)
  
  gpa_ok <- df$GPA_high == 1L
  hum    <- df$Humanities == 1L
  blue   <- df$BlueState == 1L
  
  high_pref <- gpa_ok & ( (blue & hum) | (!blue & !hum) )
  
  as.integer(high_pref)  # 0/1 predictions
}

############################################################
## ---------- 2a. K-FOLD CV FOR SIMPLE RULE ----------
############################################################

cv_simple_rule <- function(data, K = 5,
                           gpa_cut = 2.85,
                           hum_majors, blue_states,
                           seed = 123) {
  set.seed(seed)
  n <- nrow(data)
  idx <- sample.int(n)     # shuffle rows
  folds <- cut(seq_len(n), breaks = K, labels = FALSE)
  
  accs <- numeric(K)
  
  for (k in 1:K) {
    val_idx   <- idx[folds == k]
    train_idx <- setdiff(idx, val_idx)
    
    train_k <- data[train_idx, ]
    val_k   <- data[val_idx, ]
    
    y_true <- val_k$Hired
    y_pred <- rule_predict_simple(val_k,
                                  gpa_cut = gpa_cut,
                                  hum_majors = hum_majors,
                                  blue_states = blue_states)
    
    accs[k] <- mean(y_true == y_pred)
  }
  
  list(
    fold_accuracies = accs,
    mean_accuracy   = mean(accs)
  )
}

## Run CV for simple rule
cv_simple <- cv_simple_rule(train_feat,
                            K = 5,
                            gpa_cut = 2.85,
                            hum_majors = hum_majors,
                            blue_states = blue_states)

cv_simple$fold_accuracies
cv_simple$mean_accuracy

## Try different GPA cutoffs
for (cut in c(2.7, 2.8, 2.85, 2.9, 3.0)) {
  res <- cv_simple_rule(train_feat,
                        K = 5,
                        gpa_cut = cut,
                        hum_majors = hum_majors,
                        blue_states = blue_states)
  cat("GPA cutoff =", cut,
      "mean CV accuracy =", round(res$mean_accuracy, 4), "\n")
}

############################################################
## ---------- 3. GROUP-BASED RULE MODEL ----------
############################################################

cv_group_rule <- function(data, K = 5,
                          gpa_cut = 2.85,
                          hum_majors, blue_states,
                          seed = 123) {
  set.seed(seed)
  n <- nrow(data)
  idx <- sample.int(n)
  folds <- cut(seq_len(n), breaks = K, labels = FALSE)
  
  accs <- numeric(K)
  
  for (k in 1:K) {
    val_idx   <- idx[folds == k]
    train_idx <- setdiff(idx, val_idx)
    
    train_k <- data[train_idx, ]
    val_k   <- data[val_idx, ]
    
    ## Add features & group id
    train_k <- add_features(train_k, hum_majors, blue_states, gpa_cut)
    val_k   <- add_features(val_k,   hum_majors, blue_states, gpa_cut)
    
    train_k$group_id <- with(train_k,
                             paste(State, Major, GPA_high, sep = "|"))
    val_k$group_id   <- with(val_k,
                             paste(State, Major, GPA_high, sep = "|"))
    
    ## Majority vote of Hired per group in training fold
    group_prob <- tapply(train_k$Hired,
                         train_k$group_id,
                         mean)
    group_pred <- ifelse(group_prob >= 0.5, 1L, 0L)
    
    ## Map validation rows to group predictions
    m <- match(val_k$group_id, names(group_pred))
    y_pred <- group_pred[m]
    
    ## For unseen groups (NA), fall back to simple rule
    unseen <- is.na(y_pred)
    if (any(unseen)) {
      y_pred[unseen] <- rule_predict_simple(val_k[unseen, ],
                                            gpa_cut = gpa_cut,
                                            hum_majors = hum_majors,
                                            blue_states = blue_states)
    }
    
    y_true <- val_k$Hired
    accs[k] <- mean(y_true == y_pred)
  }
  
  list(
    fold_accuracies = accs,
    mean_accuracy   = mean(accs)
  )
}

## Run CV for group-based rule
cv_group <- cv_group_rule(train_feat,
                          K = 5,
                          gpa_cut = 2.85,
                          hum_majors = hum_majors,
                          blue_states = blue_states)

cv_group$fold_accuracies
cv_group$mean_accuracy

############################################################
## ---------- 3a. TRAINING ACCURACY FOR GROUP RULE ----------
############################################################

train_all <- add_features(train_feat, hum_majors, blue_states, gpa_cut = 2.85)
train_all$group_id <- with(train_all,
                           paste(State, Major, GPA_high, sep = "|"))

group_prob_all <- tapply(train_all$Hired,
                         train_all$group_id,
                         mean)
group_pred_all <- ifelse(group_prob_all >= 0.5, 1L, 0L)

m_all <- match(train_all$group_id, names(group_pred_all))
pred_train_all <- group_pred_all[m_all]

train_acc_group <- mean(pred_train_all == train_all$Hired)
train_acc_group

############################################################
## ---------- 4. PREDICT ON TEST & WRITE SORTED SUBMISSION ----------
############################################################

## test is already read & sorted by ID at the very top.
## We just need to merge State and apply the rule.

test  <- merge(test,
               loc[, c("University", "State")],
               by = "University",
               all.x = TRUE)

test$Hired <- rule_predict_simple(test,
                                  gpa_cut = 2.85,
                                  hum_majors = hum_majors,
                                  blue_states = blue_states)

submission_simple <- test[, c("ID", "Hired")]


library(randomForest) 

set.seed(101)
K <- 10
n <- nrow(train_feat)


folds <- sample(rep(1:K, length.out = n))

if (!is.factor(train_feat$Hired)) {
  train_feat$Hired <- as.factor(train_feat$Hired)
}

cv_acc_rf <- numeric(K)

for (k in 1:K) {
  cat("Fold", k, "...\n")
  
  train_idx <- which(folds != k)
  test_idx  <- which(folds == k)
  
  rf_k <- randomForest(
    Hired ~ .,
    data     = train_feat[train_idx, ],
    ntree    = 800,  
    mtry     = 2,     
    nodesize = 10    
  )
  
  pred_k <- predict(rf_k, newdata = train_feat[test_idx, ], type = "class")
  cv_acc_rf[k] <- mean(pred_k == train_feat$Hired[test_idx])
}

cv_acc_rf          
mean(cv_acc_rf)     
var(cv_acc_rf)      




## Sort submission by ID (nice & clean)
submission_simple <- submission_simple[order(submission_simple$ID), ]

write.csv(submission_simple,
          "submission_simple_rule_sorted.csv",
          row.names = FALSE)


