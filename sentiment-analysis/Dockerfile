# Use the official Python image as a base
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the app code to the container
COPY ./templates /app/templates
COPY app.py /app

# Install Flask
RUN pip install flask

# Expose port 5000
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]
