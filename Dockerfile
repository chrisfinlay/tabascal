FROM ubuntu:24.04

RUN apt-get update
RUN apt-get install -y casacore-dev libblas-dev liblapack-dev libboost-python-dev libcfitsio-dev wcslib-dev python3-pip python3-virtualenv git

WORKDIR /usr/local

RUN git clone https://github.com/chrisfinlay/tabascal.git
RUN mkdir tab_env && virtualenv tab_env && . tab_env/bin/activate && pip install -e ./tabascal/

RUN echo "source /usr/local/tab_env/bin/activate" >> /root/.bashrc

WORKDIR /data
