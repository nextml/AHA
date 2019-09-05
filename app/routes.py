import json
from flask import render_template, request
from app import app
from functools import lru_cache

@lru_cache()
def load_data():
    with app.open_resource('data.json') as data_file:
        data = json.loads(data_file.read())
    return data

@app.route('/')
@app.route('/index')
def index():
    data = load_data()[0]
    return render_template('index.html', url=data['img'], options=data['captions'], id=0, winner=-1, confidence=None, similar=None)


@app.route('/next_cartoon', methods=["POST"])
def next_cartoon():

    next_id = int(request.form['id']) + 1
    data = load_data()

    if int(next_id) >= len(data):
        next_id = 0

    data = data[next_id]
    return render_template('index.html', url=data['img'], options=data['captions'], id=next_id, winner=1, confidence=0.87, similar=["1", "2", "3"])


@app.route('/compare_captions', methods=["POST"])
def compare_captions():

    user_caption = request.form['user_caption']
    selected_caption = request.form['selected_caption']
    funnier = 0
    confidence = 0.90

    id = int(request.form['id'])
    data = load_data()[id]

    return render_template('index.html', url=data['img'], options=data['captions'], id=id, winner=funnier, confidence=0.90, similar=['c1', 'c2', 'c3'])

    return ''
