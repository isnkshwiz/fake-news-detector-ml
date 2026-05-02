import joblib

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict(text):
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]

    return "Real News" if result == 1 else "Fake News"


if __name__ == "__main__":
    user_input = input("Enter news text: ")
    print(predict(user_input))