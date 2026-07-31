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

WORKDIR /app
COPY agents.py .

RUN mkdir -p outputs

# The GPU is provided by the host at run time:
#   docker build -t agents-tutorial .
#   docker run --rm --gpus all \
#       -v /export/projects/nlp/.cache:/export/projects/nlp/.cache \
#       agents-tutorial
CMD ["python3", "agents.py"]
