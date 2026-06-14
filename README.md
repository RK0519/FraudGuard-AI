# FraudGuard AI 🛡️

.[![Live Demo](https://fraudguard-aigit-6uhvuu2qrkxybxklunqk8.streamlit.app/)].

**FraudGuard AI** is a live, Streamlit-based web application engineered to detect fraudulent credit card transactions using an optimized **Logistic Regression** pipeline. To handle extreme transactional data imbalance, the underlying architecture implements data undersampling techniques and tracks precise performance metrics to deliver reliable, end-to-end data engineering in practice.

---

## 📊 Dataset & Optimization

* **Plug-and-Play Demo Data:** A lightweight `credit_sample.csv` is included directly within this repository, allowing the Streamlit interface to run locally or on a cloud instance instantly with zero manual data configuration.
* **Full Dataset Handling:** The predictive pipeline is built to address extreme class imbalances typical of real-world financial records[cite: 1]. For benchmarking with the full-scale dataset, you can download the complete source file here: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/arockiaselciaa/creditcardcsv/data).
* **Data Engineering Layer:** Implements localized data undersampling techniques to balance highly skewed transactional data vectors prior to training the predictive Logistic Regression model.

---

## 🛠️ Tech Stack

* **Frontend & Live Deployment:** Streamlit[cite: 1]
* **Machine Learning Framework:** Scikit-Learn (Logistic Regression, Data Preprocessing, Metric Utilities)[cite: 1]
* **Data Engineering & Analysis:** Python, Pandas, NumPy[cite: 1]

---

## 🚀 How to Run Locally

Follow these steps to spin up the application on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/RK0519/FraudGuard-AI.git
cd FraudGuard-AI
