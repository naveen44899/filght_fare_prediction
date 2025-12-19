from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model & columns
model = pickle.load(open("xgboost.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "Airline": request.form["Airline"],
        "Source": request.form["Source"],
        "Destination": request.form["Destination"],
        "Journey_Day": int(request.form["day"]),
        "Journey_Month": int(request.form["month"]),
        "Total_Stops": int(request.form["stops"]),
    }

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df)[0]

    return render_template(
        "home.html",
        prediction_text=f"Predicted Flight Fare: ₹ {round(prediction, 2)}"
    )

if __name__ == "__main__":
    app.run(debug=True)


