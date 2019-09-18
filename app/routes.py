import json
from flask import render_template, request, redirect, session
from functools import lru_cache

from . import app
from . import compare_captions as comparator

# Initialize models

app.secret_key = str(hash("joes-secret_key"))

print('Initializing caption funniness comparator...')
comparator.initialize()
print("...done")

@lru_cache()
def load_data():
    with app.open_resource('data.json') as data_file:
        data = json.loads(data_file.read())
    return data

@app.route('/')
def index():
    data = load_data()
    data = [{ 'url': d['img'], 'init_cap': { 'text': d['captions'][0], 'score': '-' }, 'contest': d['contest'] } for d in data]
    return render_template('index.html', data=data)

@app.route('/compare_captions', methods=["POST"])
def compare_captions():
    caps = json.loads(request.form['caps'])
    contest = int(request.form['contest'])
    caps_raw = [x['text'] for x in caps]
    caps_raw = list(dict.fromkeys(caps_raw)) # remove duplicates

    ranks = comparator.rank_captions(caps_raw, contest)
    ranks = ranks.round(2)

    ret = []
    for k, v in ranks.to_dict().items():
        ret.append({ 'text': k, 'score': v });

    return json.dumps(ret)
