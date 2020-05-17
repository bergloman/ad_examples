FROM jupyter/tensorflow-notebook

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
RUN pip install git+https://github.com/shubhomoydas/ad_examples.git
