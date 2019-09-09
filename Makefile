debug:
	python -m flask run

deploy:
	gunicorn -w 4 app:app
