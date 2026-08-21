FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY packages/app.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .
COPY config/ config/
COPY docs/theme/ docs/theme/
COPY lib/ lib/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
