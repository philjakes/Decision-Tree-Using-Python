# Import necessary libraries
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd

# Define the dataset
data = """
FIRST,ZETECH,GOOD,YES,Y
SECOND,UON,BAD,NO,Y
FIRST,UON,GOOD,NO,N
SECOND,ZETECH,BAD,YES,N
FIRST,KU,GOOD,NO,Y
SECOND,ZETECH,BAD,NO,Y
PASS,UON,GOOD,YES,N
PASS,KU,BAD,YES,N
FIRST,KU,GOOD,NO,Y
PASS,ZETECH,BAD,YES,Y
"""

# Convert the string data to a pandas DataFrame
from io import StringIO
df = pd.read_csv(StringIO(data), header=None, names=['DEGREE', 'UNI', 'LETTER', 'EXPERIENCE', 'HIRE'])

# Convert categorical data to numerical data using Label Encoding
label_encoder = preprocessing.LabelEncoder()
df['DEGREE'] = label_encoder.fit_transform(df['DEGREE'])
df['UNI'] = label_encoder.fit_transform(df['UNI'])
df['LETTER'] = label_encoder.fit_transform(df['LETTER'])
df['EXPERIENCE'] = label_encoder.fit_transform(df['EXPERIENCE'])
df['HIRE'] = label_encoder.fit_transform(df['HIRE'])

# Split the data into features (X) and target variable (y)
X = df.drop('HIRE', axis=1)
y = df['HIRE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the model
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Display the decision tree rules
tree_rules = export_text(clf, feature_names=['DEGREE', 'UNI', 'LETTER', 'EXPERIENCE'])
print("Decision Tree Rules:\n", tree_rules)

