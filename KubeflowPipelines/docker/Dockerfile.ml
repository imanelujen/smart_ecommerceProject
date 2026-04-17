FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    pandas==2.2.0 \
    numpy==1.26.4 \
    scikit-learn==1.4.0 \
    xgboost==2.0.3 \
    joblib==1.3.2 \
    mlxtend==0.23.0 \
    kfp==2.7.0

# Copy project source
COPY Scraping/agents/     /app/agents/
COPY Scraping/orchestrator.py /app/orchestrator.py

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "-c", "print('ML container ready')"]