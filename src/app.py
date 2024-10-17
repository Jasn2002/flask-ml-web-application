from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../models/xgboost_model.pkl')

# Load the trained model once when the app starts
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Route for the form
@app.route('/')
def form():
    return render_template('form.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from the form
        hours_studied = float(request.form['hours_studied'])
        attendance = float(request.form['attendance'])
        previous_scores = float(request.form['previous_scores'])
        tutoring_sessions = float(request.form['tutoring_sessions'])

        # Prepare the input for the model
        input_data = np.array([[hours_studied, attendance, previous_scores, tutoring_sessions]])

        # Predict using the loaded model
        prediction = model.predict(input_data)[0]

        # Return the result to the user
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
     # Use the port provided by Render, or default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Set host to 0.0.0.0 to be accessible externally
    app.run(host="0.0.0.0", port=port, debug=True)