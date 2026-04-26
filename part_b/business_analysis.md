# B1. Problem Formulation

(a) ML Problem Formulation (3 marks)

Target Variable: `items_sold` — the number of items sold at a store in a given month.

Candidate Input Features:
- `store_id` — identifies the store
- `store_size` — size of the store (small/medium/large)
- `location_type` — urban, semi-urban, or rural
- `promotion_type` — type of promotion running that month (Flat Discount, BOGO, 
  Free Gift, Category-Specific Offer, Loyalty Points Bonus)
- `is_weekend` — whether the transaction occurred on a weekend
- `is_festival` — whether a festival period is active
- `competition_density` — level of nearby competition
- `month`, `day_of_week`, `year` — temporal features extracted from transaction date

Type of ML Problem: This is a supervised regression problem.

Justification: The target variable `items_sold` is a continuous numerical value. 
We have labelled historical data (past store-month records with known sales volumes) 
that can be used to train a model. The goal is to predict a quantity, not a category, 
which makes regression the appropriate problem type.

---

(b) Why Items Sold is a Better Target Variable (3 marks)

Items sold is more reliable than total sales revenue** because revenue is influenced 
by factors outside the model's control — such as product pricing, discounts, and 
seasonal price changes — which can fluctuate independently of promotion effectiveness. 
Two stores may have identical items sold but very different revenues due to price 
differences alone.

`items_sold` directly measures customer response to a promotion , making it a 
cleaner signal of promotion effectiveness.

Broader Principle: This illustrates the principle of choosing a target variable 
that is directly influenced by the decision being optimised, rather than one 
that is affected by confounding factors. In real-world ML projects, a noisy or 
proxy target variable leads to models that optimise the wrong objective. Good 
target variable selection requires domain understanding and careful thinking about 
what the business actually controls and measures.

---

(c) Alternative to a Single Global Model (2 marks)

A single global model assumes all 50 stores respond identically to promotions, 
which ignores the significant variation in location type, store size, customer 
demographics, and competition density.

Proposed Strategy: Location-Stratified or Store-Segment Models

Instead of one global model, train separate models for each store segment
(e.g. urban large, urban small, semi-urban, rural). Each segment model learns 
promotion response patterns specific to that store type.

Justification: Stores in different locations have different customer bases — 
urban customers may respond better to BOGO offers while rural customers may prefer 
Flat Discounts. A single model would average out these differences and perform 
poorly for all segments. Segmented models capture local patterns, leading to more 
accurate promotion recommendations per store type.

# B2. Data and EDA Strategy

(a) Joining Tables and Dataset Grain (4 marks)

How to join the four tables:

The four tables would be joined as follows:
- `transactions` is the base table containing store_id, date, and items_sold
- Join `store_attributes` on `store_id` to bring in store_size, location_type, 
  and competition_density
- Join `promotion_details` on `store_id` and `month` to bring in promotion_type 
  running at each store each month
- Join `calendar` on `date` to bring in is_weekend and is_festival flags

All joins would be left joins from the transactions table to preserve all 
sales records even if some metadata is missing.

Grain of the final dataset:
One row = one store × one month

Each row represents the total sales activity for a specific store in a specific 
month, along with that store's attributes, the promotion running that month, 
and the calendar context.

Aggregations before modelling:
- Sum `items_sold` per store per month
- Calculate average `competition_density` per store per month if it varies
- Extract `month`, `year`, `is_weekend` ratio, and `is_festival` flag 
  at the monthly level

---

(b) EDA Strategy (4 marks)

Analysis 1 — Promotion Type vs Average Items Sold (Bar Chart)
Plot average items_sold for each of the five promotion types. This directly 
shows which promotions drive the most sales on average, and helps identify 
whether promotion_type should be a high-priority feature or if differences 
are negligible.

Analysis 2 — Sales by Location Type (Box Plot)
Plot the distribution of items_sold grouped by location_type (urban, 
semi-urban, rural). This reveals whether location has a strong effect on 
sales volume, which would justify building separate models per location type 
or including interaction features.

Analysis 3 — Correlation Heatmap of Numerical Features
Compute correlations between items_sold and numerical features like 
competition_density, month, and is_festival. Strong correlations indicate 
which features are most predictive and should be retained. Weak correlations 
may suggest certain features can be dropped.

Analysis 4 — Sales Trend Over Time (Line Chart)
Plot monthly average items_sold over time to detect seasonality, trends, or 
sudden shifts. If strong seasonality exists, month and year become critical 
features. It also helps identify data quality issues like sudden drops or spikes.

---

(c) Handling 80% Non-Promotion Transactions (2 marks)

Impact on the model:
If 80% of transactions have no promotion, the model will be trained mostly on 
non-promotion data and may fail to learn the effect of different promotion types 
accurately. It could default to predicting average sales regardless of promotion, 
making it ineffective for the core business objective.

Steps to address it:
- Oversample promotion records using techniques like SMOTE or simple 
  random oversampling to give the model more promotion examples to learn from
- Stratified sampling during train-test split to ensure all promotion 
  types are represented proportionally in both sets
- Separate modelling — build one model for non-promotion periods and 
  another specifically for promotion periods, then combine predictions
- Feature engineering — create a binary `has_promotion` flag and include 
  it as a feature so the model explicitly learns the baseline vs promotion effect

# B3. Model Evaluation and Deployment

(a) Train-Test Split, Metrics and Interpretation (4 marks)

Train-Test Split Setup:
With three years of monthly data across 50 stores, the data should be split 
temporally — train on the first 2.5 years (months 1–30) and test on the most 
recent 6 months (months 31–36). This preserves the time ordering of the data.

Why random split is inappropriate:
This is time-ordered data — each month's sales are influenced by trends, 
seasonality, and past promotions. A random split would allow the model to 
train on future months and test on past months, causing data leakage. The 
model would appear to perform well in evaluation but fail in real deployment 
where only past data is available.

Evaluation Metrics:

- RMSE (Root Mean Squared Error): Measures average prediction error in 
  the same units as items_sold. In business terms, an RMSE of 30 means the 
  model's predictions are off by ~30 items on average. RMSE penalises large 
  errors more heavily, which matters when severely wrong recommendations 
  cause overstock or stockout.

- MAE (Mean Absolute Error): The average absolute difference between 
  predicted and actual items sold. Easier to explain to non-technical 
  stakeholders — "our model's promotion recommendations are off by X items 
  per store per month on average."

- R² (R-squared): Measures how much variance in items_sold the model 
  explains. An R² of 0.85 means the model explains 85% of the variation in 
  sales, which indicates strong predictive power. Values close to 1 are ideal.

---

(b) Explaining Different Recommendations via Feature Importance (4 marks)

The model recommends different promotions for Store 12 in December vs March 
because the input features differ significantly between these two months, 
and feature importance tells us which features drive those differences.

Investigation approach:

First, extract feature importances from the Random Forest model to identify 
the top drivers of items_sold predictions. Key features like `is_festival`, 
`month`, `is_weekend`, and `promotion_type` will have high importance scores.

Then compare the feature values for Store 12 in December vs March:
- December likely has `is_festival = 1` (holiday season) and high footfall, 
  making Loyalty Points Bonus effective for retaining high-spending customers
- March may have `is_festival = 0` and lower footfall, making Flat Discount 
  more effective at attracting price-sensitive customers

Communicating to the marketing team:
Present a simple table showing Store 12's feature values in both months 
side by side, alongside the model's predicted items_sold for each promotion 
type. Highlight that `is_festival` and `month` are the key drivers of the 
difference. Use plain language: "In December, customers are already motivated 
to buy — loyalty rewards keep them coming back. In March, a direct discount 
is needed to drive footfall."

---

(c) End-to-End Deployment Process (4 marks)

Step 1 — Save the trained model:
Use `joblib.dump(pipeline, 'promotion_model.pkl')` to serialise the full 
scikit-learn pipeline (including preprocessor and model) to disk. This ensures 
the same preprocessing steps are applied consistently at inference time.

Step 2 — Monthly data preparation:
At the start of each month, prepare a fresh input dataframe with one row per 
store containing:
- Current month, year, day_of_week features
- Store attributes (store_size, location_type, competition_density)
- Calendar flags (is_weekend ratio, is_festival)
- Each of the five promotion types as candidate inputs

Feed all five promotion options per store through the model and select the 
promotion that yields the highest predicted items_sold for each store.

Step 3 — Generate recommendations:
Load the saved model using `joblib.load('promotion_model.pkl')` and run 
`model.predict(X_new)` for all 50 stores. Output a recommendation table 
with store_id and recommended promotion_type for that month.

Step 4 — Monitoring and retraining triggers:
- Track prediction error monthly — compare predicted vs actual items_sold 
  after each month. If RMSE rises more than 15–20% above the baseline test 
  RMSE, trigger retraining.
- Monitor feature drift — check if incoming feature distributions 
  (e.g. competition_density, footfall) shift significantly from training 
  data using statistical tests. Drift indicates the model's assumptions 
  no longer hold.
- Schedule retraining — retrain the model every 6 months using the 
  most recent 2.5 years of data to keep it current with changing 
  customer behaviour and market conditions.