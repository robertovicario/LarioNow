FROM python:3.12-slim

WORKDIR /app

COPY packages/app.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .
COPY config/ config/
COPY docs/theme/ frontend/
COPY lib/ lib/
COPY models/ models/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
