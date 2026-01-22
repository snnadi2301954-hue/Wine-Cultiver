from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open('model/wine_cultivar_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Get inputs
            inputs = [float(request.form[feature]) for feature in 
                      ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'flavanoids', 'color_intensity']]
            
            # Scale inputs
            inputs_scaled = scaler.transform([inputs])
            
            # Predict
            pred_class = model.predict(inputs_scaled)[0]
            prediction = f"Cultivar {pred_class + 1}"  # +1 to match the dataset labels
        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
