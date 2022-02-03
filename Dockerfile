FROM python:3.8.0-slim

EXPOSE 5000

RUN apt-get update \
    && apt-get install gcc -y \
    && apt-get clean

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip

RUN pip install --upgrade pip setuptools wheel

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["uvicorn", "api:app", "--reload","--host", "0.0.0.0", "--port", "5000"]