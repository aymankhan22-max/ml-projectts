from flask import Flask , render_template, request
import joblib
import numpy as np 
import pandas as pd 
app = Flask(__name__)


model = joblib.load("iris_joblib")




@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        import pandas as pd

        sepal_length = float(request.form['sepal_length'])
        sepal_width  = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width  = float(request.form['petal_width'])

        features = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=[
                'sepal length (cm)',
                'sepal width (cm)',
                'petal length (cm)',
                'petal width (cm)'
            ]
        )

        prediction = model.predict(features)[0]

        
        if isinstance(prediction, (int, float)):
            prediction = ['Setosa', 'Versicolor', 'Virginica'][int(prediction)]

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return f"ERROR: {e}"


if __name__ == "__main__":
    app.run(debug=True)