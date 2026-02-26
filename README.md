# SMS Spam Detection with Naive Bayes (R)

This project builds a simple **spam vs ham SMS classifier** using the **UCI SMS Spam Collection** dataset and a classic **text mining pipeline** in R.

It follows the funnel:

**IR → NLP → IE → Data Mining**
- **IR (Information Retrieval):** download + load the dataset
- **NLP:** text cleaning (lowercasing, punctuation/numbers removal, stopwords, stemming)
- **IE (Information Extraction):** transform messages into features using a **Document-Term Matrix (DTM)** and a frequent-terms dictionary
- **Data Mining:** train and evaluate a **Naive Bayes** model (with and without Laplace smoothing)

## Dataset
- Source: UCI Machine Learning Repository — *SMS Spam Collection*  
- The script downloads the dataset automatically.

## Files
- `sms_spam_nb.R` — main script (download, preprocessing, modeling, plots, evaluation)
- `.gitignore` — ignores downloaded dataset + R session files and outputs (recommended)

## Requirements
R packages used:
- `tm`
- `SnowballC`
- `e1071`
- `gmodels`
- `wordcloud`
- `RColorBrewer`

Install them (first time only):
```r
install.packages(c("tm","SnowballC","e1071","gmodels","wordcloud","RColorBrewer"))
