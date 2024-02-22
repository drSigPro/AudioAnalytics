FROM python:3.9.18

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

## App engine stuff
# Expose port you want your app on
EXPOSE 8080

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./


# Upgrade pip 
RUN pip install -U pip

COPY requirements.txt app/requirements.txt
RUN pip install -r requirements.txt


# Run
CMD streamlit run --server.port 8080 --server.enableCORS false main.py
