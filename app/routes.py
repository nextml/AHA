import json
from flask import render_template, request
from app import app

@app.route('/')
@app.route('/index')
def index():
    with app.open_resource('data.json') as data_file:
        data = json.loads(data_file.read())
    data = data[0]
    return render_template('index.html', url=data['img'], options=data['captions'], id=0, winner=-1)

@app.route('/next_cartoon', methods=["POST"])
def next_cartoot():

    next_id = int(request.form['id']) + 1

    with app.open_resource('data.json') as data_file:
        data = json.loads(data_file.read())

    if int(next_id) >= len(data):
        next_id = 0

    data = data[next_id]
    return render_template('index.html', url=data['img'], options=data['captions'], id=next_id, winner=-1)


