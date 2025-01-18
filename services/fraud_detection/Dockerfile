# Use an official Python base image
FROM python:3.10-slim


ARG MODEL_FILE=models/best_model.pkl

# Set the environment variable, either from the argument or the default value
ENV MODEL_FILE=${MODEL_FILE}

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY api .
# COPY tests ./tests

# ENV EXPOSED_PORT=5000

# Expose the port FastAPI will run on
EXPOSE 8000


# Default command to run FastAPI
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "${EXPOSED_PORT}"]
# CMD echo ${EXPOSED_PORT}
# CMD ["echo", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", $EXPOSED_PORT]
# CMD uvicorn api:app --host 0.0.0.0 --port $EXPOSED_PORT

CMD ["gunicorn", "api:app", "-b", "0.0.0.0:8000"]