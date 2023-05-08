from flask import Flask, render_template, jsonify
import os
from flask import request

from sklearn.feature_extraction.text import TfidfVectorizer
from pyresparser import ResumeParser
from docx import Document

import nltk
import re
from ftfy import fix_text

import pandas as pd
import numpy as np

from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

csvfile = 'jobDesc_CSV.csv'
df = pd.read_csv(csvfile)


@app.route('/test')
def test():
    return render_template("index.html")

@app.route('/', methods=['GET'])
def home():
    return render_template("test.html")


@app.route("/suggest", methods=["POST"])
def suggest():
    name = request.files['resumeDoc'].filename

    if name == '':
        response = {"status": 500, "msg": "File not uploaded"}
        return jsonify(response)

    photo = request.files['resumeDoc']
    path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    photo.save(path)

    filed = path
    try:
        doc = Document()
        with open(filed, 'r') as file:
            doc.add_paragraph(file.read())
        doc.save("text.docx")
        data1 = ResumeParser('text.docx').get_extracted_data()

    except:
        data1 = ResumeParser(filed).get_extracted_data()

    resume = data1['skills']

    # matchedJobsList= []
    skills = []
    skills.append(' '.join(word for word in resume))

    skillsList = []
    for word in resume:
        skillsList.append(word)

    
    jobDesc = df['jd'].values.astype('U')
    
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
    skillV = vectorizer.fit_transform(skills)
    jobv = vectorizer.transform(jobDesc)

    nbrs = KNN(3)
    nbrs.fit(skillV, None)
    distances = nbrs.kneighbors(jobv)

    matches = pd.DataFrame(distances, columns=['confidence'])

    df['match'] = matches['confidence']
    matchedJobs = df.sort_values('match')

    matchedJobs = matchedJobs[['Job Title', 'Company Name', 'Location', 'match']].head(10).reset_index()
    matchedJobsList = matchedJobs.values.tolist()

  
    # matchedJobsList = [["0","Sfslkj","google","america","3"]]
    response = {"status": 200, "skills": skillsList, "matchedJobsList": matchedJobsList}
    return jsonify(response)


def ngrams(string, n=3):
    string = fix_text(string)  # fix text
    string = string.encode("ascii", errors="ignore").decode()  # remove non ascii chars
    string = string.lower()
    chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title()  # normalise case - capital at start of each word
    string = re.sub(' +', ' ', string).strip()  # get rid of multiple spaces and replace with a single
    string = ' ' + string + ' '  # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = []
        for x in X:
            distances = []
            for i, x_train in enumerate(self.X_train):
                distance = np.sqrt(np.sum((x - x_train) ** 2))
                distances.append((distance, self.y_train[i]))
            distances = sorted(distances)[:self.k]
            labels = [label for _, label in distances]
            predicted_label = max(set(labels), key=labels.count)
            predicted_labels.append(predicted_label)
        return predicted_labels

    def kneighbors(self, X):
        print(self.X_train.shape)
        distances = []
        for x in X:
            k = self.X_train - x
            k = k.power(2)
            distances.append(k.sum() ** .5)

        return distances


if __name__ == '__main__':
    app.run(debug=True)
