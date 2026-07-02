# Ames Housing — End-to-End ML Project
**Author:** Neža Podpeskar

## Live App
[Open the Prediction App](https://ames-housing-by-nezky.streamlit.app)

## What this project is
A complete machine learning study on the Ames Housing dataset (2,927 residential property sales in Ames, Iowa, 2006-2010), answering two questions:
- Regression: What will this home sell for? (predicts SalePrice)
- Classification: Is this a premium home? (top 25% by price)

## How to run the analysis report
1. Clone this repository
2. Install dependencies: pip install -r requirements.txt
3. Register the kernel: python -m ipykernel install --user --name ames-venv
4. Render: cd notebooks && quarto render 01_eda_and_preprocessing.qmd
5. Open notebooks/01_eda_and_preprocessing.html in your browser

## How to run the app locally
pip install -r requirements.txt
streamlit run app.py

## Architecture

```mermaid
flowchart LR
    subgraph DATA["Data & Training"]
        A["Ames Housing Dataset\n(.xls · 2,930 rows · 82 cols)"]
        B["Quarto Notebook\n(EDA & Preprocessing)"]
        C["XGBoost Regression\nreg_model.pkl"]
        D["XGBoost Classifier\nclf_model.pkl"]
        A --> B --> C & D
    end

    subgraph APP["Streamlit App — app.py"]
        T1["Home\nProject overview & key stats"]
        T2["Price & Premium Prediction\nNumber inputs → price + premium label"]
        T3["Ames's Neighborhoods\nInteractive map · 24 districts · photo cards"]
        T4["Ideal Buyer Match\nLifestyle quiz → scored neighborhood matches"]
        T5["Market Housing Insights\nEDA charts · distributions · comparisons"]
        C --> T2
        D --> T2
    end

    subgraph DEPLOY["Deployment"]
        E["GitHub\nnezapodpeskar-dotcom/\names-housing-ml-project"]
        F["Streamlit Cloud\nauto-deploys on push to main"]
    end

    subgraph USER["End User"]
        G["Browser\nDesktop & Mobile"]
    end

    DATA --> APP
    APP --> E --> F --> G
```

## Key findings
- XGBoost regression: Test R2 = 0.944, MAE = $12,353
- XGBoost classification: Test Accuracy = 96.1%, ROC-AUC = 0.9912
- No data leakage: all preprocessing fit inside a Pipeline on training data only