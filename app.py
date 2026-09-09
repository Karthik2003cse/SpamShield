from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# ── Load model & vectorizer once at startup (use 'with' so files close properly) ──
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")


@app.route("/classifier", methods=["GET", "POST"])
def classifier():
    prediction = None
    confidence = None
    email_text = ""

    if request.method == "POST":
        # .get() with a default avoids a crash if the field is missing
        email_text = request.form.get("email", "").strip()

        if not email_text:
            prediction = "Please paste a message to analyse."
        else:
            email_vector = tfidf.transform([email_text])
            raw_prediction = model.predict(email_vector)[0]

            # Handles BOTH label formats: string ("spam"/"ham") or numeric (1/0)
            is_spam = raw_prediction in ("spam", 1, "1")
            prediction = "Spam" if is_spam else "Ham"

            # Optional: confidence score, only if your model supports predict_proba
            # (e.g. LogisticRegression, MultinomialNB — not all models have this)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(email_vector)[0]
                class_index = list(model.classes_).index(raw_prediction)
                confidence = round(proba[class_index] * 100, 1)

    return render_template(
        "classifier.html",
        prediction=prediction,
        confidence=confidence,
        email_text=email_text,   # repopulates the textarea after submit
    )


if __name__ == "__main__":
    app.run(debug=True)
