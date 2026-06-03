# 🏥 Hospital Readmission Prediction Using Random Forest

## 📌 Problem Statement
The objective of this project is to predict whether a patient will be readmitted to a hospital within 30 days using a Random Forest model, and to determine the top clinical and demographic factors driving those readmissions.

* **Dataset:** [Diabetes 130-US hospitals for years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) sourced from the UCI Machine Learning Repository.

---

## ⚙️ Methodology & Experimental Design

Features were divided into three main categories and introduced to the model incrementally to evaluate if expanding the feature space yields better predictive performance.

### 1. Incremental Feature Stages
*   **Biological Factors (Base Model):** Represents baseline traits inherent to a human, such as `age`, `gender`, and `race`. The `age` variable was mapped first for optimal data representation accuracy before merging other variables.
*   **Historical Factors:** Combines the baseline model with features capturing the patient's history within the institutional healthcare setting (e.g., previous hospital interactions).
*   **Medication Factors:** Added as the final layer, working under the hypothesis that intensive medication tracking correlates with clinical acuity, which theoretically increases the likelihood of hospital readmissions.

### 2. The Core Hypothesis
> 💡 **The Forest Analogy:** The underlying experimental plan was to see if the model's predictive capability scales up cleanly with additional variables. Much like testing a real forest: if you plant more trees, does the entire ecosystem become stronger and perform better?

---

## 📈 Results & Discussion

Based on the evaluation phases, the model's overall structure showed adjustments as features were layered in, though it revealed a major technical trade-off:

*   **Classification Disparity:** While performance metrics showed movement, the model primarily improved its accuracy in predicting the *negative class*—meaning it got better at identifying patients who **would not** be readmitted (`Readmitted = No`). This runs opposite to the core objective of flagging high-risk patient returns.
*   **Performance Ceiling:** Looking closely at the final numbers, the maximum overall accuracy hovered around **56%**. Because this is a binary classification challenge, an accuracy rate this close to a standard coin flip ($50\%$) means the current feature set is not yet highly reliable for real-world deployment.

---

## 🔑 Feature Importance Insights

The Random Forest model identified the following attributes as the top three drivers of its classifications:

1. **`num_medications`:** The highest-ranked feature. This supports the initial hypothesis that a high volume of medications serves as an institutional indicator of high-severity health cases.
2. **`time_in_hospital`:** A highly logical predictor. Put simply, spending more days in a hospital bed indicates a poorer baseline condition and elevates subtle risks like nosocomial (hospital-acquired) infections.
3. **`age`:** An expected baseline element, considering that most chronic disease studies show highly documented, age-related progression patterns.

Conversely, engineered residuals like `gender_Unknown/Invalid` ranked at the very bottom. This matches data engineering expectations, as it is a residual category carrying negligible predictive weight.

---

## ⚠️ Limitations & Ethical Considerations

### 🩺 Field Insights & Unseen Variables
An accuracy ceiling of **56%** demonstrates that standard clinical records do not capture the full, complex reality of patient wellness. Drawing from practical experience working in a public hospital setting, several missing socioeconomic and structural factors strongly impact these outcomes:

*   **Socioeconomic Constraints:** Financial capability heavily influences readmission behavior regardless of illness or prescriptions. Patients frequently choose not to seek follow-up care or return to a facility simply because they cannot afford it. Instead, they may endure severe symptoms at home or transfer to lower-cost public facilities.
*   **Health Education & Literacy:** Patients from rural backgrounds or areas with limited healthcare communication may delay returning to a hospital due to institutional fear, anxiety, or conflicting cultural beliefs.
*   **Patient Attrition (Unrecorded Expirations):** The dataset lacks a post-discharge mortality track. If a patient passes away outside the hospital, they are mathematically categorized simply as "not readmitted," creating a critical blind spot that deflates the true target metrics.

### ⚖️ Ethical Challenges
*   **Sensitive Attributes:** Using demographic variables like `race` and `gender` requires extreme caution to ensure automated models do not inadvertently codify systemic healthcare disparities into medical software decisions.
*   **Data Privacy:** Utilizing historical clinical tracking underscores the continuous research dilemma of balancing open-source academic modeling with strict patient privacy rights.
