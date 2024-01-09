import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the training dataset
train_data = pd.read_csv('D:/Bharat Intern/train.csv')

# Handle missing values (Example: Impute 'Age' with median)
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

# Encode categorical variables ('Sex' and 'Embarked')
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)

# Select features and target variable
selected_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X = train_data[selected_features]
y = train_data['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
