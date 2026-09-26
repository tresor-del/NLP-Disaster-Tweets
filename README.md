# NLP Disaster Tweets

Exploratory NLP project based on the Kaggle competition "Natural Language Processing with Disaster Tweets".

The goal is to classify tweets to determine whether they refer to a real natural or human-made disaster (label `1`) or whether they are simply unrelated messages (label `0`).

## Project objective

This project demonstrates a practical text classification workflow using:

- a classic machine learning pipeline based on TF-IDF + XGBoost,
- a deep learning pipeline based on embeddings and LSTM,
- notebooks for analysis, cleaning, preprocessing, modeling, and evaluation.

The main purpose is to compare two approaches on a binary text classification problem:

- `0` = non-disaster tweet,
- `1` = disaster-related tweet.

---

## Data

This project uses tweet data from the original Kaggle disaster tweet dataset. 

The goal is to classify whether a tweet is related to a real disaster event or not. In practice, the data is used in several stages:

- raw text samples for model training and evaluation,
- cleaned and normalized text for preprocessing,
- feature representations for classical machine learning models,
- padded token sequences for recurrent neural network models,
- labels and predictions used for evaluation.

---

## Model approaches

### 1) Classical baseline: BoW / TF-IDF + XGBoost

This branch uses:

- text cleaning,
- tokenization / preprocessing,
- TF-IDF vectorization,
- an XGBoost classifier.

---

### 2) Deep learning approach: embeddings + LSTM

This branch explores a deep learning approach:

- loading pre-trained embeddings (`wiki-words`),
- word vectorization with zero-padding,
- an LSTM network with a `sigmoid` output for binary classification,
- evaluation through accuracy, loss, and confusion matrix.


---

## Repository structure

```text
nlp_disaster_tweets/
├── data/
│   ├── processed/
│   ├── raw/
│   └── predictions/
├── model/
│   ├── bow-tfidf/
│   └── rnn-lstm/
├── notebooks/
│   ├── bow-tfidf/
│   └── rnn-lstm/
├── .venv/
├── README.md
└── .gitignore
```

---


## Tools and dependencies

The project is built around standard Python NLP libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `tensorflow` / `keras`
- `matplotlib`
- `seaborn`
- `joblib`
- `kagglehub`
- `nltk`


