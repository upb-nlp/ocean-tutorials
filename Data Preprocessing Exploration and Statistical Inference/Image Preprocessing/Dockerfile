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
COPY image_preprocessing.py .

RUN mkdir -p outputs/image_preprocessing

# Sample images ship with scikit-image; the Olivetti faces dataset (Lab 5)
# is downloaded automatically on first run.
CMD ["python3", "image_preprocessing.py"]
