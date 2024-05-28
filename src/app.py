from flask import Flask, request, render_template, jsonify
from prediction import preprocess, load_models, predict
from datafetch import fetch_twitter_data

app = Flask(__name__)

vectoriser, LRmodel = load_models()

@app.route('/')
def home():
    return render_template('index.html')

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

@app.route('/live', methods=['GET', 'POST'])
def live():
    if request.method == 'GET':
        return render_template('newTwitter.html')
    elif request.method == 'POST':
        try:
            keyword = request.json['text']
            texts = fetch_twitter_data(keyword)
            if texts:
                positive_texts = [] 
                negative_texts = []  
                total_positive_count = 0
                total_negative_count = 0
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
                return jsonify({
                    'positive': total_positive_percentage,
                    'negative': total_negative_percentage,
                    'positive_texts':positive_texts,
                    'negative_texts':negative_texts
                })
            else:
                return jsonify({"error": "No tweets found for the given hashtag."})
        except Exception as e:
            return jsonify({"error": str(e)}) 
        
@app.route('/result')
def result():
    return render_template('newResults.html')

if __name__ == '__main__':
    app.run(debug=True)
