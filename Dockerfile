#FROM amazonlinux:2 as build
FROM public.ecr.aws/lambda/python:3.7

#RUN set -x \
#    && python3.7 -m venv /app



ENV PATH="/app/.bin:${PATH}"

ENV BOGUS 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /app
#ENV PYTHONPATH "${PYTHONPATH}:/app/"

WORKDIR /app

#ARG DEVEL=no

RUN yum clean all && yum update -y
#RUN yum groupinstall -y "Development Tools" && \
 #   yum clean all 
#yum install -y python3-pip && \
RUN yum install -y git && \
    yum clean metadata && \
    yum install -y shadow-utils.x86_64 -y && \
    yum clean metadata && \
    yum install -y gcc && \
    yum clean all


RUN pip --no-cache-dir --disable-pip-version-check install --upgrade pip setuptools wheel

COPY requirements.txt ./


RUN pip install  --no-cache-dir --disable-pip-version-check --no-deps \
    -r requirements.txt -t .

RUN  pip3 --no-cache-dir --disable-pip-version-check install 'git+https://github.com/facebookresearch/fvcore'

RUN pip3 --no-cache-dir --disable-pip-version-check install detectron2==0.6 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html  

COPY fold1_model_0315229.pth ./

COPY app.py ./


CMD ["app.handler"]