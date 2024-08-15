from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and vectorizer
tfidf = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# Porter Stemmer initialization
ps = PorterStemmer()

# Text Preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Define the main route ("/") for rendering the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Define the predict route to handle form submission and classification
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_sms = request.form['sms']  # Get the SMS/email text from the form

        # 1. Preprocess the input text
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize the transformed input
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict if the message is spam or not
        result = model.predict(vector_input)[0]

        # 4. Return the result (Spam or Not Spam)
        if result == 1:
            prediction = "Spam"
        else:
            prediction = "Not Spam"

        # Render the result on the webpage
        return render_template('result.html', prediction=prediction)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
