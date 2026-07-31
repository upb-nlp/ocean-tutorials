FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    python3-venv \
    && apt-get clean

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/
RUN pip install --upgrade pip wheel
RUN pip install -r /tmp/requirements.txt

# spaCy English model (Lab 3: POS, NER, dependency parsing)
RUN python -m spacy download en_core_web_sm

WORKDIR /app
COPY nlp_foundations.py .

# NLTK data and the 20 Newsgroups corpus are downloaded automatically on first run.
CMD ["python3", "nlp_foundations.py"]
