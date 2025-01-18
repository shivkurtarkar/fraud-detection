# Use an official Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY api .
# COPY tests ./tests

ENV EXPOSED_PORT=8000

# Expose the port FastAPI will run on
EXPOSE $EXPOSED_PORT


# Default command to run FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "${EXPOSED_PORT}"]
# CMD ["uvicorn", "app", "--host", "0.0.0.0", "--port", "5000"]