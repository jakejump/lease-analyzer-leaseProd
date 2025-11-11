FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libheif1 \
    libde265-0 \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r backend/requirements.txt

# Expose port and run app
EXPOSE 8000
RUN echo "✅ Dockerfile successfully built and running!"
ENV OCR_AGENT=unstructured_pytesseract
CMD ["sh", "-c", "mkdir -p /data/storage /data/temp && ln -sfn /data/storage /app/storage && ln -sfn /data/temp /app/temp && uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]


