FROM python:3.8.0-slim

ENV API_KEY=bcqoieyqp98DAHJBABJBy3498ypiuqhriuqy984
ENV API_KEY_NAME=access_token

EXPOSE 47122

RUN apt-get update \
    && apt-get install gcc -y \
    && apt-get clean

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python", "-m", "uvicorn", "api:app", "--reload","--host", "0.0.0.0", "--port", "47122"]