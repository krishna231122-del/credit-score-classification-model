import pandas as pd
import joblib
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")
new_data = {
    "Age": 30,
    "Income": 50000,
    "Loan Amount": 20000,
    "Marital Status": "Single"
}
df = pd.DataFrame([new_data])
df = pd.get_dummies(df, drop_first=True)
df = df.reindex(columns=columns, fill_value=0)
prediction = model.predict(df)
print("Prediction:", prediction)
