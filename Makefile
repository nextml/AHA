debug:
	python -m flask run

deploy:
	gunicorn -t 120 -b '0.0.0.0:5000' -w 4 app:app
