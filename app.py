from tabulate import tabulate
from flask import Flask, request, render_template
import clean_text as ct
import bigram as bg
import model as md

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        '''
        For rendering results on HTML GUI
        '''
        int_features = str(request.form.get("description"))

        string = ct.textPreProcessing(int_features)

        vocab_dict , arr = bg.textProcessing(string)
        
        tf = bg.computeTF(vocab_dict,arr)
        idf = bg.computeIDF([vocab_dict])
        tfidf_bg = bg.computeTfidf(tf,idf)

        tfidf = md.textprocessing(string)

        headers = [['Rank', 'Key Bigrams', 'Keywords']]
        bigrams = tfidf_bg
        keywords = tfidf.index.tolist()

        for i in range(25):
            headers.append([int(i)+1, bigrams[i], keywords[i].upper()])
        
        table = tabulate(headers, tablefmt = 'html')
        return table
    except:
        return "<h1>NOT ENOUGH KEYWORDS IN THE DESCRIPTION</h1><iframe src=\"https://giphy.com/embed/Jt4sQOFEh29Ob8KAxg\" width=\"240\" height=\"135\" frameBorder=\"0\" class=\"giphy-embed\" allowFullScreen></iframe>"

if __name__ == "__main__":
    app.run(debug=True)