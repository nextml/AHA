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
def next_cartoot():

    next_id = int(request.form['id']) + 1
    data = load_data()

    if int(next_id) >= len(data):
        next_id = 0

    data = data[next_id]
    return render_template('index.html', url=data['img'], options=data['captions'], id=next_id, winner=1, confidence=0.87, similar=["1", "2", "3"])


