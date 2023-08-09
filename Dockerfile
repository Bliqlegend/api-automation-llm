FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    firefox-esr

RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.30.0/geckodriver-v0.30.0-linux64.tar.gz \
    && tar -zxf geckodriver-v0.30.0-linux64.tar.gz -C /usr/local/bin \
    && chmod +x /usr/local/bin/geckodriver

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "main.py", "--server.port", "80"]
