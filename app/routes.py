import json
from flask import render_template, request, redirect, session
from app import app
from functools import lru_cache
from app import compare_captions

# Initialize models

app.secret_key = 'joe'

print('Initializing caption funniness comparator.....')
compare_captions.initialize()
# comparator.initialize()

@lru_cache()
def load_data():
    with app.open_resource('data.json') as data_file:
        data = json.loads(data_file.read())
    return data

@app.route('/')
def index():
    data = load_data()[0]
    return render_template('index.html', url=data['img'], options=data['captions'], id=0, winner=-1, confidence=None, similar=None, c1=None, c2=None)


@app.route('/next_cartoon', methods=["POST"])
def next_cartoon():

    next_id = int(request.form['id']) + 1
    data = load_data()

    if int(next_id) >= len(data):
        next_id = 0

    data = data[next_id]
    return render_template('index.html', url=data['img'], options=data['captions'], id=next_id, winner=-1, confidence=None, similar=None, c1=None, c2=None)


@app.route('/compare_captions', methods=["POST"])
def compare_captions():

    id = int(request.form['id'])
    data = load_data()[id]

    user_caption = request.form['user_caption']
    caps = user_caption.split('\n')
    if 'caption' not in session:
        session['captions'] = caps
    else:
        session['captions'].append(caps)

    ranks = compare_captions.rank_captions(session['captions'], data['contest'])
    print(ranks)

    return render_template('index.html', url=data['img'], options=ranks['caption'].values, id=id, winner='', confidence='', similar='', c1='')
