# Use an official Python runtime as a parent image
FROM python:3.6-slim

# Set the working directory to /app
WORKDIR /app

ADD . /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN apt-get update && apt-get install -y libgtk2.0-dev

EXPOSE 8080

# Run app.py when the container launches
CMD ["python", "app.py"]