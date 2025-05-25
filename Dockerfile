FROM python:3.11.12-bullseye

RUN apt-get update

WORKDIR /usr/local

RUN git clone https://github.com/chrisfinlay/tabascal.git
RUN pip install -e ./tabascal/

WORKDIR /data
