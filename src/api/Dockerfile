# Dockerfile pour src/api

FROM python:3.11-slim

WORKDIR /app

# Copie des requirements
COPY requirements/ requirements/

# Copie du code
COPY src/ src/

# Mise à jour d'apt-get et installation des dépendances Python
RUN apt-get update \
    && apt-get install python3-pip -y \
    && pip3 install --no-cache-dir -r requirements/api.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]