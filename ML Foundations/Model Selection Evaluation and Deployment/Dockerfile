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
COPY model_selection.py .

RUN mkdir -p outputs/model_selection

# optuna (Bayesian search) and shap (interpretability) are optional — the
# script degrades gracefully if they are absent, but the image installs them.
CMD ["python3", "model_selection.py"]
