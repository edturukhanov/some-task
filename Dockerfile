FROM python:3.7

WORKDIR /tmp
COPY requirements.txt /tmp

# Install dependencies
RUN pip install -r requirements.txt

# Run the application:
