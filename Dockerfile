FROM continuumio/anaconda3

RUN /opt/conda/bin/conda install -y pytorch torchvision cpuonly -c pytorch
RUN /opt/conda/bin/conda install -y -c conda-forge regex
RUN /opt/conda/bin/conda install -y -c conda-forge uwsgi

ADD requirements.txt /requirements.txt
RUN /opt/conda/bin/pip install -r /requirements.txt
RUN /opt/conda/bin/python -m spacy download en_core_web_lg
RUN /opt/conda/bin/python -m spacy download en

COPY app /var/www/app
ENV FLASK_APP=/var/www/app
EXPOSE 5000
CMD ["/opt/conda/bin/python", "-m", "flask", "run"]
