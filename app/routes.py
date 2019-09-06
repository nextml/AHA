import json
from flask import render_template, request
from app import app
from functools import lru_cache
from app import compare_captions as comparator
from app import get_similar as similarity 

# Initialize models

print('Initializing caption funniness comparator.....')
comparator.initialize()
print('Initializing caption similarity comparator.....')
similarity.initialize()


@lru_cache()
def load_data():
    with app.open_resource('data.json') as data_file:
        data = json.loads(data_file.read())
    return data

@app.route('/')
@app.route('/index')
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
    contest = data['contest']

    user_caption = request.form['user_caption']
    selected_caption = request.form['selected_caption']
    info = comparator.compare_captions(user_caption, selected_caption, contest)
    winner = 0 if info['funnier'] == 1.0 else 1
    confidence = int(100 * info['proba'])

    caps = data['captions']
    similar = similarity.get_most_similar(user_caption, caps)

    return render_template('index.html', url=data['img'], options=data['captions'], id=id, winner=winner, confidence=confidence, similar=similar, c1=user_caption, c2=selected_caption)
