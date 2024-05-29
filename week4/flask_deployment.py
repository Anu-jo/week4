import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()
breast_cancer_df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
breast_cancer_df['target'] = breast_cancer.target
X=breast_cancer_df.drop(columns=['target'],axis=1)
y=breast_cancer_df['target']
#print(breast_cancer_df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
with open('linear_model.pkl', 'wb') as file:
    pickle.dump(model, file)
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model
with open('linear_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Ensure the input data is in the correct format
    features = [data['features']]
    prediction = model.predict(features)
    return jsonify({'Cancer_target': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)


