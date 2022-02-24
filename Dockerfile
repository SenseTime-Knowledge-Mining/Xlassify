FROM python:3.7.1

WORKDIR /xlassify

RUN python -m pip install --upgrade pip \
	&& pip install --no-cache-dir xlassify

ENTRYPOINT ["xlassify"]