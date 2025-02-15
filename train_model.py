import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

train_df = pd.read_csv("training.csv")


X = train_df.drop(columns=['prognosis'])  
y = train_df['prognosis'] 

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


with open("disease_model.pkl", "wb") as file:
    pickle.dump((model, label_encoder, X.columns.tolist()), file)

print("Model trained and saved as disease_model.pkl")
