from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        features = [
            float(request.form["attendance"]),
            float(request.form["midsem"]),
            float(request.form["assignment"]),
            float(request.form["gpa"]),
            float(request.form["quiz"])
        ]
        pipeline = PredictPipeline()
        prediction = pipeline.predict(features)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
