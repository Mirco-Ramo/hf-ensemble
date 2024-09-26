FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04 AS base

ARG GIT_TOKEN

RUN : "${GIT_TOKEN:?Build argument needs to be set and non-empty.}"

ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-transport-https ca-certificates gnupg software-properties-common wget 

RUN wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add - 
RUN apt-get install -y --no-install-recommends cmake build-essential python3 python3-pip python3-packaging \
                                               libboost-all-dev zlib1g-dev libbz2-dev protobuf-compiler \
                                               libprotobuf-dev openssl libssl-dev libgoogle-perftools-dev git git-lfs


RUN git clone https://$GIT_TOKEN@github.com/Mirco-Ramo/hf-ens-dist.git /opt/ens-dist

RUN git lfs install

# dev tools
RUN apt-get install less unzip curl wget lsof vim -y
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && ./aws/install && rm -rf aws*

RUN pip install -r /opt/ens-dist/requirements.txt -U

WORKDIR /opt/ens-dist
