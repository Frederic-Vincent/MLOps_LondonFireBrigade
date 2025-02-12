# Dockerfile pour src/frontend

FROM python:3.11-slim

WORKDIR /app

# Copie des requirements
COPY requirements/ requirements/

# Copie du code
COPY src/ src/

# Mise à jour d'apt-get et installation des dépendances Python
RUN apt-get update \
    && apt-get install python3-pip -y \
    && pip3 install --no-cache-dir -r requirements/frontend.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8501

CMD ["streamlit", "run", "src/frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]