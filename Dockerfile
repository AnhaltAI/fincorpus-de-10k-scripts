FROM python:3.10

# install python requirements
COPY code/requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
RUN python3 -m spacy download de_core_news_lg

# copy data
COPY FULL_CLI /FULL_CLI
COPY code .

CMD python3 lang-detection.py
