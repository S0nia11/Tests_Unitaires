from src.preprocessing import preprocess
from src.modeling import build_model

def main():
    text = [
        "Earthquake strikes again!",
        "Sunny day in Paris.",
        "Massive fire in the forest.",
        "I love pizza."
    ]
    labels = [1, 0, 1, 0]

    print("Texte prétraité :")
    processed = [preprocess(t) for t in text]
    print(processed)

    print("\nEntraînement du modèle...")
    model = build_model()
    model.fit(processed, labels)

    print("\nPrédictions :")
    predictions = model.predict(processed)
    print(predictions)

if __name__ == "__main__":
    main()
