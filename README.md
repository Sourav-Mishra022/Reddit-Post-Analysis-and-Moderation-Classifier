# Reddit Post Analysis and Moderation Classifier

## 📘 Overview
This project analyzes Reddit post data to identify whether a post is likely to be **removed by moderators** based on textual and metadata features.  
It combines **Exploratory Data Analysis (EDA)**, **SQL-style querying**, and **Machine Learning (ML)** to uncover patterns behind post removals on the *r/DataIsBeautiful* subreddit.

---

## 🎯 Objectives
- Perform **EDA** and derive structured insights using SQL-style queries.
- Identify correlations between **post metrics** (score, comments, awards) and **removal status**.
- Clean and preprocess text for model training using **tokenization, stopword removal, stemming**, and **TF-IDF vectorization**.
- Train and evaluate **Decision Tree** and **Random Forest** models to classify post removal.
- Optimize performance using **SMOTE**, **class weighting**, **threshold tuning**, and **ensemble learning**.

---

## 🧩 Dataset
**File:** `r_dataisbeautiful_posts.csv`  
**Source:** Reddit API / Pushshift dataset  
**Size:** 193,000 posts  
**Key Columns:**
| Column | Description |
|--------|--------------|
| `title` | Post title text |
| `score` | Reddit upvotes score |
| `num_comments` | Number of comments |
| `total_awards_received` | Total awards received |
| `created_utc` | UTC timestamp of post |
| `removed_by` | Moderator removal status (target variable) |

**Target Variable:**
- `removed_by` → Converted to binary:  
  `1 = removed`, `0 = visible`

---

## 🧮 Exploratory Data Analysis (EDA)
- Conducted EDA using **Pandas** and **PandaSQL** to query data with `GROUP BY`, `HAVING`, and `CASE` statements.
- Analyzed correlations using **Pearson’s correlation** between numeric features and post removal status.
- Generated visual insights using **WordClouds** to explore common terms in removed vs. non-removed posts.

---

## 🧹 Data Preprocessing
- Missing value imputation for numeric columns.
- Text preprocessing steps:
  - Lowercasing and punctuation removal  
  - URL and special character removal  
  - Stopword removal (`nltk.stopwords`)  
  - Stemming (`PorterStemmer`)  
  - TF-IDF vectorization (word and char n-grams)
- Created metadata features: `text_len`, `word_count`, and `hour` of post creation.

---

## ⚙️ Model Development
### Models Trained:
1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **Random Forest + SMOTE (balanced)**
4. **Random Forest + Logistic Regression Ensemble**

### Techniques Used:
- **SMOTE** (Synthetic Minority Oversampling) for handling class imbalance  
- **Class Weights** to improve minority recall  
- **Threshold Optimization** for maximizing F1-score  
- **Cross-validation (Stratified K-Fold)** for robust performance evaluation  

---

## 📈 Model Evaluation

| Model | F1 (Removed) | Recall | Precision | Accuracy |
|--------|---------------|---------|------------|-----------|
| Decision Tree | 0.33 | 0.29 | 0.37 | 0.86 |
| Random Forest | 0.41 | 0.46 | 0.38 | 0.89 |
| SMOTE + Random Forest | 0.45 | 0.55 | 0.38 | 0.88 |
| **Ensemble (RF + Logistic)** | **0.59** | **0.90** | 0.44 | **0.92** |

**Best Model:**  
✅ **Ensemble of Random Forest & Logistic Regression**  
- F1-score: **0.59**  
- Recall: **0.90**  
- Macro F1: **0.77**  
- Weighted F1: **0.92**

---

## 🧠 Key Insights
- Posts with **low score** or **few comments** are more likely to be removed.  
- **Certain phrases and patterns** (e.g., “OC”, “Map”, “Dataset”) are common among non-removed posts.  
- **Metadata features** (time of posting, number of awards) influence moderation likelihood.  
- Balancing the dataset using **SMOTE** significantly improved recall on the minority class (removed posts).  

---

## 🛠️ Tools & Technologies
| Category | Tools |
|-----------|--------|
| Programming | Python |
| Data Analysis | Pandas, NumPy, PandaSQL |
| Visualization | Matplotlib, WordCloud |
| NLP | NLTK, TF-IDF (Scikit-learn) |
| ML Models | DecisionTree, RandomForest, LogisticRegression |
| Balancing | SMOTE (Imbalanced-learn) |
| Evaluation | F1-score, Recall, Precision, Confusion Matrix |
| Environment | Google Colab / Jupyter Notebook |

---

## 🚀 Results
The final model achieved a **Threshold-optimized F1-score of 0.59** and **Recall of 0.90**,  
demonstrating strong ability to detect posts that would likely be moderated.

---

## 📂 Repository Structure
```
├── r_dataisbeautiful_posts.csv         # Raw dataset
├── reddit_moderation_classifier.ipynb  # Main notebook
├── models/                             # (optional) saved trained models
├── README.md                           # Project documentation
```

---

## 📚 Future Improvements
- Incorporate **subreddit-specific linguistic patterns**
- Include **BERT embeddings** for semantic context (optional extension)
- Experiment with **LightGBM/XGBoost** for higher precision
- Deploy as a **moderation support tool** using Streamlit

---

## 🧑‍💻 Author
**Sourav Mishra**  
Machine Learning & Data Science Enthusiast  
📧 [Your Email Here]  
🔗 [Your LinkedIn Here]
