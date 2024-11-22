# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy requirements and application files
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY model.pkl model.pkl
COPY scaler.pkl scaler.pkl

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
