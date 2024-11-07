import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r"UCI_Credit_Card.csv")

X = data[['LIMIT_BAL', 'SEX']].values
y = data['default.payment.next.month'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(class_weight='balanced')
model.fit(X_scaled, y)


limit_bal = int(input("Enter the LIMIT_BAL: "))
sex = int(input("Enter the SEX (1 for male, 2 for female): "))


card = np.array([[limit_bal, sex]])
card_scaled = scaler.transform(card)


default_prediction = model.predict(card_scaled)[0]
default_probability = model.predict_proba(card_scaled)[0][1]


if default_prediction == 1:
    print("Default")
else:
    print("No Default")
