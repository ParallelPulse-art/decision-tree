"""
Run this script first to train the model and generate model.pkl
Usage: python train_model.py --data titanic.csv
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='titanic.csv')
args = parser.parse_args()

df = pd.read_csv(args.data)
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
df = df[features + ['survived']].copy()

age_median = df['age'].median()
fare_median = df['fare'].median()

df['age'] = df['age'].fillna(age_median)
df['fare'] = df['fare'].fillna(fare_median)
df['embarked'] = df['embarked'].fillna('S')

le_sex = LabelEncoder()
le_emb = LabelEncoder()
df['sex'] = le_sex.fit_transform(df['sex'])
df['embarked'] = le_emb.fit_transform(df['embarked'])

X = df[features]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
print(classification_report(y_test, model.predict(X_test)))

with open('model.pkl', 'wb') as f:
    pickle.dump({
        'model': model, 'le_sex': le_sex, 'le_emb': le_emb,
        'age_median': age_median, 'fare_median': fare_median
    }, f)

print("✅ model.pkl saved!")
