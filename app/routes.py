from flask import render_template
from app import app

@app.route('/')
@app.route('/index')
def index():
    url = "https://raw.githubusercontent.com/nextml/caption-contest-data/master/contests/info/adaptive/657/657.jpg?raw=true"
    options = [
        "option 1",
        "option 2",
        "option 3",
        "option 4",
        "option 5"
    ]
    return render_template('index.html', url=url, options=options)
