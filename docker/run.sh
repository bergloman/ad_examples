#!/bin/bash

# cd ..

# docker run -it -v $PWD:/home/dev bergloman:ad_examples bash

# docker run -it -v ${PWD}:/home/jovyan/work bergloman:ad_examples bash
# docker run -it -v $PWD:/home/jovyan/work -e GRANT_SUDO=yes bergloman:ad_examples bash


# docker run -it -v $PWD:/home/jovyan/work -e GRANT_SUDO=yes tensorflow/tensorflow:2.1.1 bash

docker-compose -f docker/docker-compose-flat.yml up
# docker-compose -f docker/docker-compose.yml up

# docker run -it -v $PWD:/tf/jovyan/work bergloman:ad_examples bash
# docker run -it -v ${PWD}:/tf/jovyan/work bergloman:ad_examples bash


# iforest
# bash ./aad.sh datacenter123 35 1 0.03 7 1 0 0 512 0 1 1

# iforest streaming
# bash ./aad.sh datacenter123 35 1 0.03 7 1 0 1 512 0 1 1

# loda
# bash ./aad.sh datacenter123 35 1 0.03 13 1 0 0 512 0 1 1
