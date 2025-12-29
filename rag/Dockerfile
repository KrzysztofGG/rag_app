FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first to avoid wheels issues
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Download the model using the module directly
# Adding a check to ensure spacy is present
RUN python -m spacy download pl_core_news_sm

COPY . .

CMD ["fastapi", "run", "main.py", "--port", "8000"]