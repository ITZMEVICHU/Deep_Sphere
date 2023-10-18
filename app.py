# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('model.pkl')




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            try:
                # Read the uploaded dataset
                data = pd.read_csv(uploaded_file)
                # Assuming the dataset has 'earnings' and 'earning_potential' columns

                # Make predictions using the model
                predictions = model.predict(data[['earnings', 'earning_potential']])

                # Add the predictions to the dataset
                data['prefarable_spending_limit'] = predictions

                # Convert the dataset to HTML table
                result_table = data.to_html(classes='table table-bordered table-hover', index=False)
            except Exception as e:
                result_table = str(e)
        else:
            result_table = "Please upload a valid CSV file."

    return render_template('index.html', result=result_table)

if __name__ == '__main__':
    app.run(debug=True)
