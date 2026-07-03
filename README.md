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

## User Journey

```mermaid
journey
    title User Journey — Ames Housing ML App
    section Arrive
        Land on Home tab: 5: User
        Read project overview & key stats: 4: User
        See model summary cards: 4: User
    section Predict
        Go to Price & Premium Prediction tab: 5: User
        Set quality, area, bathrooms, garage: 5: User
        Pick year built & neighbourhood: 5: User
        Click Predict button: 5: User
        View estimated sale price: 5: User
        View Premium Home verdict: 5: User
        Read Investment Insights: 4: User
    section Explore
        Go to Ames's Neighborhoods tab: 5: User
        Browse interactive map: 5: User
        Read premium photo cards: 4: User
        Compare budget & mid-range districts: 4: User
    section Match
        Go to Ideal Buyer Match tab: 5: User
        Select buyer persona: 5: User
        Set lifestyle preferences: 5: User
        View top neighbourhood matches: 5: User
        Review Top 5 comparison table: 4: User
    section Insights
        Go to Market Housing Insights tab: 4: User
        Explore EDA charts & distributions: 4: User
        Compare neighbourhoods by price: 4: User
```

## Build & Deployment Timeline

```mermaid
gantt
    title Build & Deployment Timeline
    dateFormat  YYYY-MM-DD
    axisFormat  %b %Y

    section Phase 1 — Data
        Load & inspect dataset          :done, p1a, 2025-02-01, 7d
        Clean & impute missing values   :done, p1b, after p1a, 7d
        Feature engineering             :done, p1c, after p1b, 5d

    section Phase 2 — Modelling
        EDA & visualisations            :done, p2a, after p1c, 7d
        XGBoost regression (price)      :done, p2b, after p2a, 5d
        XGBoost classifier (premium)    :done, p2c, after p2b, 5d
        Quarto analysis report          :done, p2d, after p2c, 4d

    section Phase 3 — App
        Streamlit skeleton & routing    :done, p3a, after p2d, 5d
        Home & EDA tabs                 :done, p3b, after p3a, 4d
        Predict tab + results display   :done, p3c, after p3b, 6d
        Neighborhoods tab + map         :done, p3d, after p3c, 7d
        Ideal Buyer Match tab           :done, p3e, after p3d, 8d

    section Phase 4 — Deploy
        GitHub repo setup               :done, p4a, after p3e, 2d
        Streamlit Cloud deployment      :done, p4b, after p4a, 2d
        Mobile responsiveness fixes     :done, p4c, after p4b, 4d
        Final polish & AI reflection    :done, p4d, after p4c, 3d
```

## Key findings
- XGBoost regression: Test R2 = 0.944, MAE = $12,353
- XGBoost classification: Test Accuracy = 96.1%, ROC-AUC = 0.9912
- No data leakage: all preprocessing fit inside a Pipeline on training data only

My final models predict home prices within about $12,000 on average and correctly classify premium homes 96% of the time on unseen test data — and because all preprocessing was fit only on training data inside a Pipeline, these results are trustworthy, not inflated by data leakage.