FROM registry.baidubce.com/paddlepaddle/paddle:2.4.1-gpu-cuda10.2-cudnn7.6-trt7.0 
ENV LANG=en_US.utf8
ENV LANG=C.UTF-8

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

RUN mkdir /opt/program/
RUN pip3 install -i https://mirror.baidu.com/pypi/simple networkx==2.3 flask gevent gunicorn boto3
RUN pip3 install -i https://mirror.baidu.com/pypi/simple paddlenlp onnx onnxconverter_common onnxruntime-gpu nvgpu

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /dev/stdout /var/log/nginx/access.log
RUN ln -sf /dev/stderr /var/log/nginx/error.log

COPY model /opt/program
RUN ls /opt/program

WORKDIR /opt/program