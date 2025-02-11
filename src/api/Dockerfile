FROM python:3.11-slim

WORKDIR /app

# Copie des fichiers de dépendances
COPY requirements.txt setup.py README.md ./

# Copie du code
COPY src/ src/

# Mise à jour d'apt-get
RUN apt-get update 

# Installation des mises à jour de Python
RUN apt-get install python3-pip -y 

# Installation des dépendances
RUN pip3 install --no-cache-dir -r requirements.txt

# Installation du paquet en mode développement
RUN pip3 install -e .

EXPOSE 8000

CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]