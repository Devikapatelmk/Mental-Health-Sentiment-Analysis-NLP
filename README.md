# 🧠 Mental Health Sentiment Analysis Using NLP

An NLP-based machine learning project for analyzing text statements and classifying them into different mental-health-related categories.

The project uses **Natural Language Processing (NLP)**, **TF-IDF feature extraction**, and multiple machine learning algorithms to classify user-provided text into one of **7 categories**.

An interactive **Streamlit web application** is also included for testing the trained model with custom text.

---

## 📌 Project Information

- **Project Type:** Major Project - NLP
- **Batch:** Data Science MITM
- **Project Title:** Sentiment Analysis for Mental Health Monitoring
- **Domain:** Natural Language Processing / Machine Learning
- **Number of Classes:** 7
- **Best Model:** Logistic Regression
- **Final Test Accuracy:** 74.12%

---

## 🎯 Project Objective

The objective of this project is to build an NLP-based text classification system capable of identifying patterns associated with different mental-health-related categories from textual statements.

The system performs:

- Dataset preprocessing
- Exploratory Data Analysis
- Text cleaning
- Text analysis and visualization
- Target encoding
- TF-IDF feature extraction
- Machine learning model training
- Model comparison
- Detailed model evaluation
- Custom text prediction
- Model saving and verification
- Streamlit web application development

> **Important:** This project is developed for educational purposes only and must not be used as a medical or psychological diagnostic system.

---

# 📊 Dataset

The project uses the **Combined Data.csv** mental health text dataset.

### Original Dataset Shape

```text
53,043 rows
3 columns
```

### Original Columns

```text
Unnamed: 0
statement
status
```

The unnecessary `Unnamed: 0` column was removed during dataset preparation.

### Main Features

| Column | Description |
|---|---|
| `statement` | Text statement provided in the dataset |
| `status` | Mental-health-related category |
| `cleaned_statement` | Preprocessed version of the statement |
| `label` | Encoded target category |

---

# 🧠 Mental Health Categories

The dataset contains **7 categories**:

```text
Anxiety
Bipolar
Depression
Normal
Personality disorder
Stress
Suicidal
```

### Label Encoding

| Label | Category |
|---:|---|
| 0 | Anxiety |
| 1 | Bipolar |
| 2 | Depression |
| 3 | Normal |
| 4 | Personality disorder |
| 5 | Stress |
| 6 | Suicidal |

---

# 🔍 Exploratory Data Analysis

After removing missing values for text-based analysis:

```text
Records analyzed: 52,681
```

### Original Class Distribution

| Category | Count | Percentage |
|---|---:|---:|
| Normal | 16,351 | 30.83% |
| Depression | 15,404 | 29.04% |
| Suicidal | 10,653 | 20.08% |
| Anxiety | 3,888 | 7.33% |
| Bipolar | 2,877 | 5.42% |
| Stress | 2,669 | 5.03% |
| Personality disorder | 1,201 | 2.26% |

The dataset is therefore **imbalanced**, with Normal being the largest class and Personality disorder being the smallest.

### Text Statistics

```text
Average words per statement : 113.16
Median words per statement  : 62
Minimum words               : 1
Maximum words               : 6,300
```

The class imbalance ratio was approximately:

```text
13.61
```

---

# 🧹 Text Preprocessing

Several preprocessing techniques were applied to improve the quality of the text before feature extraction.

The preprocessing pipeline includes:

- Converting text to lowercase
- Removing URLs
- Removing HTML content
- Removing email addresses
- Removing user mentions
- Processing hashtags
- Expanding common contractions
- Removing punctuation
- Removing special characters
- Removing unnecessary numbers
- Removing extra whitespace
- Removing stopwords
- Preserving important negation words
- Lemmatization

Important negation words such as:

```text
no
not
nor
never
```

were preserved because they can significantly affect the meaning of a sentence.

### Dataset Cleaning Results

```text
Original Dataset Size : 53,043
Final Dataset Size    : 50,970
Records Removed       : 2,073
Reduction             : 3.91%
```

Missing statements and duplicate statements were removed.

---

# 📚 Text Analysis

After preprocessing:

```text
Total Cleaned Statements : 50,970
Total Words              : 2,670,559
Vocabulary Size          : 50,065
```

### Top Frequent Words

Some of the most frequently occurring words were:

```text
not
like
feel
want
know
life
get
time
no
even
would
people
year
day
thing
really
cannot
one
going
think
```

The most frequent word was:

```text
not
```

with:

```text
86,644 occurrences
```

This demonstrates the importance of preserving negation during preprocessing.

---

# 🔢 Feature Extraction

The cleaned text was transformed into numerical features using:

## TF-IDF Vectorization

Configuration:

```text
Maximum Features : 30,000
N-Gram Range     : (1, 2)
```

This means the vectorizer considers both:

- Unigrams
- Bigrams

Examples include:

```text
self
harm
self harm
going away
friend week
```

### TF-IDF Matrix

```text
Training Matrix : (40,776, 30,000)
Testing Matrix  : (10,194, 30,000)
```

Training matrix sparsity:

```text
99.81%
```

---

# ✂️ Train-Test Split

The dataset was divided using an **80:20 train-test split**.

```text
Total Samples    : 50,970

Training Samples : 40,776
Testing Samples  : 10,194
```

Stratification was used to preserve the class distribution.

---

# 🤖 Machine Learning Models

Four machine learning algorithms were trained and compared.

1. Logistic Regression
2. Linear Support Vector Machine
3. Random Forest
4. Multinomial Naive Bayes

---

# 📈 Model Comparison

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| **Logistic Regression** | **74.12%** | **0.7407** | **0.6854** |
| Linear SVM | 73.52% | 0.7315 | 0.6815 |
| Random Forest | 68.46% | 0.6610 | 0.5171 |
| Multinomial Naive Bayes | 63.66% | 0.6147 | 0.4201 |

---

# 🏆 Best Model

The best-performing model was:

```text
Logistic Regression
```

### Final Performance

```text
Accuracy             : 74.12%
Weighted Precision   : 0.7502
Weighted Recall      : 0.7412
Weighted F1 Score    : 0.7407

Macro Precision      : 0.6648
Macro Recall         : 0.7180
Macro F1 Score       : 0.6854
```

Logistic Regression was selected primarily using **Macro F1 Score**, because the dataset contains significant class imbalance.

---

# 📋 Class-Wise Performance

| Category | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Normal | 0.8717 | 0.9125 | **0.8916** |
| Anxiety | 0.7466 | 0.8301 | **0.7861** |
| Bipolar | 0.7038 | 0.7460 | **0.7243** |
| Suicidal | 0.6432 | 0.7043 | **0.6724** |
| Depression | 0.7688 | 0.5887 | **0.6668** |
| Personality disorder | 0.4762 | 0.6145 | **0.5366** |
| Stress | 0.4433 | 0.6296 | **0.5203** |

### Best Performing Category

```text
Normal
F1 Score: 0.8916
```

### Weakest Performing Category

```text
Stress
F1 Score: 0.5203
```

---

# 🔄 Confusion Matrix Analysis

The detailed evaluation showed that some categories contain overlapping linguistic patterns.

The most common misclassification was:

```text
Actual Depression → Predicted Suicidal
724 samples
```

Another major confusion was:

```text
Actual Suicidal → Predicted Depression
365 samples
```

This indicates substantial language overlap between Depression and Suicidal statements in the dataset.

---

# ✅ Prediction Performance

The final model was evaluated on:

```text
10,194 test samples
```

Results:

```text
Correct Predictions   : 7,556
Incorrect Predictions : 2,638

Correct Percentage    : 74.12%
Incorrect Percentage  : 25.88%
```

Average prediction confidence:

```text
63.75%
```

---

# 💬 Custom Text Prediction

A custom prediction function was developed that performs the following pipeline:

```text
User Text
    ↓
Text Preprocessing
    ↓
TF-IDF Transformation
    ↓
Logistic Regression
    ↓
Predicted Category
    ↓
Confidence Score
    ↓
Probability Distribution
```

### Example

Input:

```text
I have been feeling nervous and worried for several days and I cannot relax.
```

Output:

```text
Predicted Category : Anxiety
Confidence         : 77.98%
```

Probability distribution:

```text
Anxiety               77.98%
Depression              8.86%
Stress                  6.24%
Suicidal                4.06%
Normal                  2.14%
Personality disorder    0.38%
Bipolar                 0.34%
```

---

# 💾 Saved Model Files

The final trained system was saved using Joblib.

```text
mental_health_model.pkl
tfidf_vectorizer.pkl
label_encoder.pkl
```

### File Purpose

| File | Purpose |
|---|---|
| `mental_health_model.pkl` | Trained Logistic Regression classifier |
| `tfidf_vectorizer.pkl` | Converts cleaned text into TF-IDF features |
| `label_encoder.pkl` | Converts encoded predictions into category names |

The saved files were reloaded and tested successfully.

### Verification Result

```text
Verification Status: PASSED
```

Verification example:

```text
Input:
I feel anxious and worried and I cannot relax or sleep properly.

Prediction:
Anxiety

Confidence:
89.21%
```

---

# 🌐 Streamlit Web Application

A simple interactive web application was developed using **Streamlit**.

The application allows users to:

- Enter custom text
- Analyze the statement
- View the predicted category
- View prediction confidence
- View preprocessed text
- View probabilities for all seven categories
- Visualize category probabilities using a chart
- View model information

---

# 📁 Project Structure

```text
Mental_Health_Sentiment_Analysis/
│
├── app.py
│
├── README.md
│
├── requirements.txt
│
│
└── mental_health_model_files/
    │
    ├── mental_health_model.pkl
    ├── tfidf_vectorizer.pkl
    └── label_encoder.pkl
```

The Google Colab notebook used for model development can also be included in the repository.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Mental-Health-Sentiment-Analysis-NLP.git
```

Move into the project directory:

```bash
cd Mental-Health-Sentiment-Analysis-NLP
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Streamlit Application

Run:

```bash
streamlit run app.py
```

If the `streamlit` command is not recognized, use:

```bash
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

in your browser if it does not open automatically.

---

# 📦 Requirements

The main libraries used in this project are:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
joblib
streamlit
```

A simple `requirements.txt` can contain:

```text
streamlit
scikit-learn
joblib
nltk
pandas
numpy
```

---

# 🛠️ Technologies Used

- Python
- Google Colab
- Visual Studio Code
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF
- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes
- Random Forest
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- GitHub

---

# 📌 Project Workflow

```text
Dataset
   ↓
Dataset Understanding
   ↓
Exploratory Data Analysis
   ↓
Missing Value & Duplicate Removal
   ↓
Text Preprocessing
   ↓
Text Analysis & Visualization
   ↓
Target Encoding
   ↓
Train-Test Split
   ↓
TF-IDF Feature Extraction
   ↓
Model Training
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Detailed Evaluation
   ↓
Custom Text Prediction
   ↓
Model Saving & Verification
   ↓
Streamlit Web Application
```

---

# 📝 Project Tasks

The project was implemented in multiple stages:

### Task 1
Project setup, library import and dataset extraction.

### Task 2
Dataset understanding and data quality analysis.

### Task 3
Exploratory Data Analysis.

### Task 4
Text preprocessing and cleaning.

### Task 5
Text analysis and visualization.

### Task 6
Target encoding.

### Task 7
Train-test split and TF-IDF feature extraction.

### Task 8
Machine learning model training and comparison.

### Task 9
Detailed evaluation of the best model.

### Task 10
Custom text prediction system.

### Task 11
Model saving, reloading and verification.

### Final Application
Streamlit-based interactive prediction interface.

---

# 🔬 Key Findings

- Logistic Regression produced the best overall performance.
- It achieved **74.12% test accuracy**.
- Normal was the best classified category with an F1 score of **0.8916**.
- Anxiety also achieved strong performance with an F1 score of **0.7861**.
- Stress was the most difficult category to classify.
- Depression and Suicidal statements showed considerable linguistic overlap.
- Class imbalance affected the performance of minority categories.
- TF-IDF with unigram and bigram features worked effectively for this multi-class NLP problem.
- More computationally expensive models did not necessarily provide better results. Random Forest, for example, performed below Logistic Regression and Linear SVM on the sparse TF-IDF representation.

---

# ⚠️ Limitations

The project has several limitations:

1. The dataset is significantly imbalanced.
2. Some mental-health-related categories use similar vocabulary.
3. TF-IDF primarily captures statistical word patterns and has limited understanding of deeper context.
4. Short or ambiguous statements can produce low-confidence predictions.
5. The model can sometimes be strongly influenced by explicit category-related keywords.
6. Predictions depend heavily on the quality and labeling of the training dataset.
7. The system must not be interpreted as a clinical diagnostic tool.

---

# 🚀 Future Scope

Future improvements may include:

- BERT
- RoBERTa
- DistilBERT
- Sentence Transformers
- Deep learning architectures
- Improved class balancing
- Hyperparameter optimization
- Larger and more diverse datasets
- Explainable AI techniques
- Improved confidence calibration
- More advanced contextual text representations
- Deployment as a secure web application

Transformer-based models could potentially capture contextual relationships more effectively than traditional TF-IDF features.

---

# 🎓 Conclusion

This project demonstrates the complete development of an NLP-based multi-class text classification system for mental-health-related text.

After preprocessing and analyzing more than **50,000 statements**, TF-IDF was used to convert textual information into numerical features. Four machine learning algorithms were trained and evaluated.

**Logistic Regression** achieved the best overall performance with:

```text
Accuracy    : 74.12%
Weighted F1 : 0.7407
Macro F1    : 0.6854
```

The trained model was successfully saved, reloaded, verified, and integrated into an interactive Streamlit application.

The project demonstrates practical applications of **NLP, text preprocessing, feature engineering, machine learning, model evaluation, model persistence, and application development** while also highlighting the challenges of class imbalance and overlapping language in mental-health-related text.

---

# ⚠️ Disclaimer

This project is intended **strictly for educational and research purposes**.

The predicted categories are based only on statistical patterns learned from the training dataset.

The application:

- Does **not** diagnose mental health conditions.
- Does **not** replace qualified healthcare professionals.
- Should **not** be used for clinical decision-making.
- Should **not** be used for emergency assessment.

---

# ⭐ Repository

If you find this project useful for learning NLP and machine learning, consider giving the repository a ⭐.

### Project

**Sentiment Analysis for Mental Health Monitoring**
