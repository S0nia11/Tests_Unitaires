FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt stopwords wordnet
CMD ["python", "main.py"]
