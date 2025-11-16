# EMIPredict AI - Intelligent Financial Risk Assessment Platform

This project is a complete FinTech web application that provides a data-driven solution for financial risk assessment. It uses machine learning to predict EMI (Equated Monthly Installment) eligibility (a **Classification** task) and the maximum affordable EMI amount (a **Regression** task).

The platform is built on a dataset of 400,000 financial records and integrates **MLflow** for experiment tracking and a **Streamlit** web application for real-time predictions.

## 🚀 Key Features
* **Dual-Problem Solution:** Solves both Classification (EMI Eligibility) and Regression (Max EMI Amount).
* **Advanced Feature Engineering:** Creates financial ratios like DTI, Savings Ratio, and a custom Financial Risk Score.
* **MLflow Integration:** Logs all model experiments, parameters, and metrics for comparison and versioning.
* **Streamlit Web App:** A multi-page interactive dashboard for real-time risk assessment.
* **Production-Ready:** The project is structured with separate scripts for training and for the application, following best practices.

## 🛠️ How to Run This Project in VS Code

### Step 1: Set Up Your Environment
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/BaraShowCode/YOUR-REPO-NAME.git](https://github.com/BaraShowCode/YOUR-REPO-NAME.git)
    cd YOUR-REPO-NAME
    ```
2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    ```
3.  **Activate the Environment:**
    * **Windows:** `.\venv\Scripts\activate`
    * **Mac/Linux:** `source venv/bin/activate`
4.  **Install All Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Run the Project
You will need **3 separate terminals** in VS Code.

#### **Terminal 1: Start the MLflow UI**
This lets you watch your models train and compare their results.
```bash
mlflow ui