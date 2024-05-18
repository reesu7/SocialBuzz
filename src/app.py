from flask import Flask, request, render_template
from prediction import preprocess, load_models, predict
from datafetch import fetch_twitter_data

# Initialize the Flask app
app = Flask(__name__)

# Load the models
vectoriser, LRmodel = load_models()

# Redirects to the landing page
@app.route('/')
def home():
    return render_template('index.html')

# Redirects to the page where user inputs data
# and gets the sentiment of the input text
@app.route('/typing', methods=['GET', 'POST'])
def typing():
    if request.method == 'GET':
        return render_template('typing.html')
    elif request.method == 'POST':
        try:
            text = request.form['text']
            processed_text = preprocess([text])  
            df = predict(vectoriser, LRmodel, processed_text) 
            sentiment = df['sentiment'].iloc[0]
            return render_template('typing.html', sentiment=sentiment)
        except Exception as e:
            error = str(e)
            return render_template('typing.html', error=error)

# Redirects to the page where user inputs a hashtag
# and gets the sentiment of the tweets related to the hashtag
@app.route('/live', methods=['GET', 'POST'])
def live():
    if request.method == 'GET':
        return render_template('live.html')
    elif request.method == 'POST':
        try:
            text = request.form['text']
            texts = fetch_twitter_data(text)
            if texts:
                total_positive_count = 0
                total_negative_count = 0
                positive_texts = [] 
                negative_texts = []  
                vectoriser, LRmodel = load_models()
                for text in texts:
                    processed_text = preprocess([text])  
                    df = predict(vectoriser, LRmodel, processed_text)
                    sentiment = df['sentiment'].iloc[0]
                    if sentiment == 'Positive':
                        total_positive_count += 1
                        positive_texts.append(text)
                    elif sentiment == 'Negative':
                        total_negative_count += 1
                        negative_texts.append(text)
                total_texts = len(texts)
                total_positive_percentage = (total_positive_count / total_texts) * 100
                total_negative_percentage = (total_negative_count / total_texts) * 100

                return render_template('live.html', 
                                       total_positive_percentage=total_positive_percentage,
                                       total_negative_percentage=total_negative_percentage,
                                       positive_texts=positive_texts,
                                       negative_texts=negative_texts)
            else:
                return render_template('live.html', error="No tweets found for the given hashtag.")
        except Exception as e:
            error = str(e)
            return render_template('live.html', error=error)


if __name__ == '__main__':
    app.run(debug=True)
