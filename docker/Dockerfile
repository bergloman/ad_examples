FROM tensorflow/tensorflow:2.1.1-jupyter

# RUN apt-get update && apt-get install -y bzip2 cron git gzip mc nano tar vim zip && apt-get autoremove -y && apt-get clean

COPY ./requirements.txt /requirements.txt

RUN pip install -r /requirements.txt
RUN pip install git+https://github.com/shubhomoydas/ad_examples.git
