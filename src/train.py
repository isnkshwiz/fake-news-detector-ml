from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib

from data_preprocessing import load_data, preprocess
from vectorizer import create_vectorizer


def train():
    df = preprocess(load_data())

    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = create_vectorizer()

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    from sklearn.model_selection import cross_val_score

    print("\n--- Cross Validation ---")

    for name, model in models.items():
        scores = cross_val_score(model, X_train_vec, y_train, cv=5)
        print(f"{name} CV Score: {scores.mean()}")

    best_model = None
    best_score = 0

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        print(f"\n{name} Accuracy: {acc}")
        print(classification_report(y_test, preds))

        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_test, preds)

        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        if acc > best_score:
            best_score = acc
            best_model = model

    # save best model
    joblib.dump(best_model, "models/model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")


if __name__ == "__main__":
    train()