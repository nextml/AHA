import json
from flask import render_template, request, redirect, session
from functools import lru_cache

from . import app
from . import compare_captions as comparator

# Initialize models

app.secret_key = str(hash("joes-secret_key"))

print('Initializing caption funniness comparator...')
# comparator.initialize()
print("...done")

@lru_cache()
def load_data():
    with app.open_resource('data.json') as data_file:
        data = json.loads(data_file.read())
    return data

@app.route('/')
def index():
    data = load_data()
    data = [{ 'url': d['img'], 'init_cap': { 'text': d['captions'][0], 'score': '-' } } for d in data]
    return render_template('index.html', data=data)

@app.route('/compare_captions', methods=["POST"])
def compare_captions():

    id = int(request.form['id'])
    data = load_data()[id]

    user_caption = request.form['user_caption']

    if (user_caption is None or user_caption.strip().strip('\n') == '') and len(data['captions']) == 1:
        ret = get_init_options(data)
        return render_template('index.html', url=data['img'], options=ret, id=id)

    caps = user_caption.split('\n')
    caps = [c for c in caps if c != '' and c != '\n']
    if 'captions' not in session:
        session['captions'] = caps
    else:
        session['captions'] = session['captions'] + caps

    ranks = comparator.rank_captions(session['captions'], data['contest'])
    ranks = ranks.round(2)

    new = []
    for k, v in ranks.to_dict().items():
        if k in caps:
            new.append(k)

    return render_template('index.html', url=data['img'], options=ranks.to_dict(), id=id, new=new)
