import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,roc_auc_score

df = pd.read_csv(r"C:\Users\Johnnel\Desktop\TIP folder\1st year 2nd sem\applied machine learning\activity\A4\diabetes+130-us+hospitals+for+years+1999-2008\diabetic_data.csv")

#CLEANING
#REPLACE
df.replace('?', np.nan, inplace=True)

#DROP
high_miss_val=["weight", "max_glu_serum", "A1Cresult", "medical_specialty", "payer_code"]
low_val=["encounter_id", "patient_nbr"]
part_drop=["race", "diag_1", "diag_2", "diag_3"]
df = df.drop_duplicates(subset='patient_nbr', keep='first')

df.drop(columns=high_miss_val,inplace=True, errors="ignore")
df.drop(columns=low_val,inplace=True, errors="ignore")
df.dropna(subset=part_drop, inplace=True)
df=df.drop_duplicates()

#TARGET
df["readmitted"] = df["readmitted"].apply(lambda x: 0 if x =="NO" else 1)

#FEATURE
#Biological factors
age_map = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95}
df["age"] = df['age'].map(age_map)

bio_f= ['gender','race']
bio = df[bio_f]
bf = pd.get_dummies(bio, drop_first=True)

#Historical factors
hist = ['number_inpatient', 'number_emergency','time_in_hospital', 'num_medications']
his = df[hist]
hx = pd.get_dummies(his, drop_first=True)

#Medication
medic = ['insulin', 'metformin', 'diabetesMed']
med = df[medic]
md = pd.get_dummies(med, drop_first=True)

#BIOLOGICAL BASE MODEL
#SPLITTING
y = df["readmitted"]
X = pd.concat([df[['age']], bf], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#MODEL
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

#METRICS
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
prc = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

#HEATMAP
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Predicted: No', 'Predicted: Yes'],
            yticklabels=['Actual: No', 'Actual: Yes'])
plt.title("Actual vs. Predicted: Base Model")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

#Base METRICS
print(f"Base Acc: {acc:.4f}")
print(f"Base Prc: {prc:.4f}")
print(f"Base Rec: {rec:.4f}")
print(f"Base AUC: {auc:.4f}")

#BIO BASE MODEL w/ HX
#SPLITTING
y = df["readmitted"]
X = pd.concat([df[['age']], bf, hx], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#MODEL
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

#METRICS
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
prc = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

#HEATMAP
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Predicted: No', 'Predicted: Yes'],
            yticklabels=['Actual: No', 'Actual: Yes'])
plt.title("Actual vs. Predicted: w/ History Model")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

#Hist METRICS
print(f"Hist Acc: {acc:.4f}")
print(f"Hist Prc: {prc:.4f}")
print(f"Hist Rec: {rec:.4f}")
print(f"Hist AUC: {auc:.4f}")

#BIO BASE MODEL w/ HX & Med
#SPLITTING
y = df["readmitted"]
X = pd.concat([df[['age']], bf, hx, md], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#MODEL
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

#METRICS
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
prc = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

#HEATMAP
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Predicted: No', 'Predicted: Yes'],
            yticklabels=['Actual: No', 'Actual: Yes'])
plt.title("Actual vs. Predicted: w/ History & Med Model")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

#Med METRICS
print(f"Med Acc: {acc:.4f}")
print(f"Med Prc: {prc:.4f}")
print(f"Med Rec: {rec:.4f}")
print(f"Med AUC: {auc:.4f}")

#FEAT COMPARISON
importances = rf.feature_importances_
feature_names = X.columns
feature_series = pd.Series(importances, index=feature_names)

#FEAT COMPARISON PLOT
feature_series.sort_values().plot(kind="barh", color="b")
plt.title('Feature Importance: Which Factor Matters Most?')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()
