##############################################
# SMS Spam Classification (UCI) — Full Script
# IR → NLP → IE → Data Mining + Visualizations
##############################################

## 0) Packages (install once, then comment install line)
# install.packages(c("tm","SnowballC","e1071","gmodels","wordcloud","RColorBrewer"))

library(tm)
library(SnowballC)
library(e1071)
library(gmodels)
library(wordcloud)
library(RColorBrewer)

set.seed(123)  # reproducibility

## 1) IR: Download + load dataset
url <- "https://archive.ics.uci.edu/static/public/228/sms%2Bspam%2Bcollection.zip"
download.file(url, destfile = "sms_spam_collection.zip", mode = "wb")
unzip("sms_spam_collection.zip")

sms <- read.table("SMSSpamCollection",
                  sep = "\t", header = FALSE,
                  col.names = c("label", "message"),
                  quote = "", comment.char = "",
                  stringsAsFactors = FALSE)

sms$label <- factor(sms$label)
cat("Rows:", nrow(sms), "\n")
print(table(sms$label))

## 2) Quick EDA / Visualizations
# 2.1 Class distribution
barplot(table(sms$label),
        main = "SMS Class Distribution",
        ylab = "Count")

# 2.2 Message length distribution by class
sms$msg_len <- nchar(sms$message)
boxplot(msg_len ~ label, data = sms,
        main = "Message Length by Class",
        ylab = "Number of characters")

# 2.3 Histogram of message lengths (overall)
hist(sms$msg_len,
     main = "Message Length Distribution",
     xlab = "Characters",
     breaks = 40)

## 3) Split into train/test (random, unbiased)
idx <- sample(seq_len(nrow(sms)), size = 0.75 * nrow(sms))
sms_raw_train <- sms[idx, ]
sms_raw_test  <- sms[-idx, ]

cat("\nTrain label proportions:\n")
print(prop.table(table(sms_raw_train$label)))
cat("\nTest label proportions:\n")
print(prop.table(table(sms_raw_test$label)))

## 4) NLP: Create corpus + clean text (train/test separately to avoid leakage)
make_clean_corpus <- function(text_vec) {
  corp <- Corpus(VectorSource(text_vec))
  corp <- tm_map(corp, content_transformer(tolower))
  corp <- tm_map(corp, removeNumbers)
  corp <- tm_map(corp, removePunctuation)
  corp <- tm_map(corp, stripWhitespace)
  corp <- tm_map(corp, removeWords, stopwords("en"))
  corp <- tm_map(corp, stemDocument)  # optional; keep for textbook-like pipeline
  return(corp)
}

sms_corpus_train <- make_clean_corpus(sms_raw_train$message)
sms_corpus_test  <- make_clean_corpus(sms_raw_test$message)

## 5) IE: Build DTM + dictionary of frequent terms (from TRAIN only)
sms_dtm_train_full <- DocumentTermMatrix(sms_corpus_train)
sms_dtm_test_full  <- DocumentTermMatrix(sms_corpus_test)

# frequent words threshold
freq_terms <- findFreqTerms(sms_dtm_train_full, 5)

# rebuild DTMs using dictionary
sms_train <- DocumentTermMatrix(sms_corpus_train, control = list(dictionary = freq_terms))
sms_test  <- DocumentTermMatrix(sms_corpus_test,  control = list(dictionary = freq_terms))

cat("\nNumber of features after dictionary filtering:", ncol(sms_train), "\n")

## 6) Visualizations: Word frequencies + word clouds (from TRAIN)
train_term_freq <- sort(colSums(as.matrix(sms_train)), decreasing = TRUE)

# 6.1 Top 20 words overall (train)
top20 <- head(train_term_freq, 20)
barplot(top20,
        las = 2,
        main = "Top 20 Words (Train)",
        ylab = "Total count")

# 6.2 Wordcloud overall (train)
set.seed(123)
wordcloud(words = names(train_term_freq),
          freq = train_term_freq,
          min.freq = 40,
          random.order = FALSE,
          max.words = 150,
          colors = brewer.pal(8, "Dark2"))

# 6.3 Wordcloud spam vs ham (train) using raw text (easy + meaningful)
spam_train <- subset(sms_raw_train, label == "spam")
ham_train  <- subset(sms_raw_train, label == "ham")

set.seed(123)
wordcloud(spam_train$message,
          max.words = 60,
          scale = c(3, 0.6),
          random.order = FALSE,
          colors = brewer.pal(8, "Reds"))

set.seed(123)
wordcloud(ham_train$message,
          max.words = 60,
          scale = c(3, 0.6),
          random.order = FALSE,
          colors = brewer.pal(8, "Blues"))

## 7) Convert counts to categorical Yes/No for Naive Bayes
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}

sms_train_nb <- apply(sms_train, MARGIN = 2, convert_counts)
sms_test_nb  <- apply(sms_test,  MARGIN = 2, convert_counts)

## 8) Data Mining: Train Naive Bayes + Evaluate
# 8.1 Without Laplace smoothing
sms_classifier <- naiveBayes(sms_train_nb, sms_raw_train$label)
sms_test_pred  <- predict(sms_classifier, sms_test_nb)

cat("\nConfusion Matrix (No Laplace):\n")
CrossTable(sms_test_pred, sms_raw_test$label,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c("predicted", "actual"))

# 8.2 With Laplace smoothing (recommended)
sms_classifier2 <- naiveBayes(sms_train_nb, sms_raw_train$label, laplace = 1)
sms_test_pred2  <- predict(sms_classifier2, sms_test_nb)

cat("\nConfusion Matrix (Laplace = 1):\n")
CrossTable(sms_test_pred2, sms_raw_test$label,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c("predicted", "actual"))

## 9) Efficiency Metrics: Accuracy, Precision, Recall, F1 (for spam)
compute_metrics <- function(pred, actual, positive = "spam") {
  cm <- table(predicted = pred, actual = actual)
  
  TP <- cm[positive, positive]
  TN <- sum(diag(cm)) - TP
  FP <- sum(cm[positive, ]) - TP
  FN <- sum(cm[, positive]) - TP
  
  accuracy  <- (TP + TN) / sum(cm)
  precision <- TP / (TP + FP)
  recall    <- TP / (TP + FN)
  f1        <- 2 * precision * recall / (precision + recall)
  
  out <- c(accuracy = accuracy, precision = precision, recall = recall, f1 = f1)
  return(out)
}

cat("\nMetrics (No Laplace):\n")
print(compute_metrics(sms_test_pred, sms_raw_test$label))

cat("\nMetrics (Laplace = 1):\n")
print(compute_metrics(sms_test_pred2, sms_raw_test$label))

## 10) Optional: Save objects/results
# saveRDS(sms_classifier2, "sms_nb_model.rds")
# write.csv(data.frame(pred = sms_test_pred2, actual = sms_raw_test$label),
#           "sms_predictions.csv", row.names = FALSE)

cat("\nDone.\n")
