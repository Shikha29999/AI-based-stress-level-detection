import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('D:/Stress Level Detection/app/stress_data.csv')


# Encode all categorical columns
le_emotion = LabelEncoder()
le_sleep = LabelEncoder()
le_appetite = LabelEncoder()
le_remedy = LabelEncoder()

df['Emotion'] = le_emotion.fit_transform(df['Emotion'])
df['Sleep Quality'] = le_sleep.fit_transform(df['Sleep Quality'])
df['Appetite'] = le_appetite.fit_transform(df['Appetite'])
df['Remedy'] = le_remedy.fit_transform(df['Remedy'])

# Features and target
X = df[['Emotion', 'Sleep Quality', 'Appetite']]
y = df['Remedy']

# Train decision tree model
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Save model and encoders
joblib.dump(clf, 'remedy_tree_model.pkl')
joblib.dump(le_emotion, 'le_emotion.pkl')
joblib.dump(le_sleep, 'le_sleep.pkl')
joblib.dump(le_appetite, 'le_appetite.pkl')
joblib.dump(le_remedy, 'le_remedy.pkl')

print("Decision tree model trained and saved successfully.")
