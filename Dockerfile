FROM python:3.7

ENV OJ_ENV production

ADD . /app
WORKDIR /app

HEALTHCHECK --interval=5s --retries=3 CMD python2 /app/deploy/health_check.py

RUN apt-get update && apt-get upgrade && apt-get install curl ca-certificates gnupg

RUN curl https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -

RUN apt-get install -y nginx openssl curl unzip supervisor libjpeg-dev zlib1g-dev postgresql   freetype2-demos libfreetype6 libfreetype6-dev && \
    pip install --no-cache-dir -r /app/deploy/requirements.txt && apt-get purge

ENTRYPOINT /app/deploy/entrypoint.sh
