from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("artifacts/model.pkl", "rb"))

def compute_risk_score(attendance, midsem, assignment, gpa, quiz):
    """
    Normalized weighted score out of 100.
    Max marks: attendance=100, midsem=20, assignment=5, quiz=5, gpa=10
    Weights:   attendance=30%, midsem=35%, assignment=15%, quiz=10%, gpa=10%
    """
    score = (
        (attendance / 100) * 30 +
        (midsem     /  20) * 35 +
        (assignment /   5) * 15 +
        (quiz       /   5) * 10 +
        (gpa        /  10) * 10
    )
    return round(score, 2)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    risk = None

    if request.method == "POST":
        attendance = float(request.form["attendance"])
        midsem     = float(request.form["midsem"])
        assignment = float(request.form["assignment"])
        gpa        = float(request.form["gpa"])
        quiz       = float(request.form["quiz"])

        # Feature order must match training column order
        features = [attendance, midsem, assignment, gpa, quiz]

        pred = model.predict([features])[0]
        prediction = "Pass" if pred == 1 else "Fail"

        score = compute_risk_score(attendance, midsem, assignment, gpa, quiz)

        if score >= 70:
            risk = "Low Risk"
        elif score >= 50:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

    return render_template("index.html", prediction=prediction, risk=risk)


if __name__ == "__main__":
    app.run(debug=True)