
FROM ubuntu:24.04

RUN apt-get update &&     apt-get install -y     python3-pip     python3-venv     && apt-get clean

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/
RUN pip install --upgrade pip wheel
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 
RUN pip install -r /tmp/requirements.txt
