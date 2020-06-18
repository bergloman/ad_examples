FROM python:3.6.10

# RUN apt-get update && apt-get install -y bzip2 cron git gzip mc nano tar vim zip && apt-get autoremove -y && apt-get clean

COPY ./requirements.txt /requirements.txt

RUN pip install -r /requirements.txt
# WORKDIR /home/jovyan
# RUN pip install git+https://github.com/shubhomoydas/ad_examples.git
