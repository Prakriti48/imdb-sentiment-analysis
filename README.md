📘 IMDB Movie Review Sentiment Analysis

A Streamlit web application that uses a pretrained ReLU-based RNN model to classify IMDB movie reviews as Positive or Negative.

🚀 Project Overview

This project demonstrates how to use a pretrained Simple RNN model in TensorFlow/Keras to perform sentiment analysis on movie reviews.
The app:

Accepts user input

Preprocesses the text

Feeds it to the loaded RNN model

Displays the predicted sentiment and confidence score

The UI is built using Streamlit, making the model easily accessible via a browser interface.

📂 Project Structure
├── simple_rnn_imdb_model.h5      # Trained RNN model
├── app.py                        # Streamlit application script
└── README.md                     # Documentation

🧠 Model Details

Dataset: IMDB reviews (Keras built-in dataset)

Architecture: Simple RNN

Activation Function: ReLU

Output Layer: Sigmoid

Task: Binary Sentiment Classification

🛠️ Requirements

Install dependencies using:

pip install tensorflow streamlit numpy

▶️ How to Run the Application

Place your model file:

simple_rnn_imdb_model.h5


in the same directory as app.py.

Run the Streamlit application:

streamlit run app.py


Your browser will open automatically at:

http://localhost:8501

📝 Code Explanation
🔹 Loading IMDB Vocabulary
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

🔹 Loading the Pretrained Model
model = load_model('simple_rnn_imdb_model.h5')

🔹 Preprocessing User Input

Tokenizes

Converts to word index

Pads to length 500

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

🔹 Prediction
prediction = model.predict(preprocessed_input)
sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"

🖼️ Streamlit User Interface

Text area for user input

Button to classify

Displays sentiment + probability score

st.title("IMDB Movie Review Sentiment Analysis")

✨ Features

✔ Real-time sentiment prediction
✔ Clean and simple UI
✔ Uses pretrained Keras RNN model
✔ Easy to extend or integrate into larger applications

📢 Future Improvements

Add LSTM / GRU models

Integrate with HuggingFace Transformer models

Add dataset visualization

Deploy on Streamlit Cloud / AWS

🤝 Contributing

Pull requests and suggestions are welcome!

📄 License

This project is free to use for educational purposes.
