# docker build -t els . 
# docker run --rm -v $(pwd):/context -w /context -it els /bin/bash
# >  /tmp/extract_loudest_section/gen/bin/extract_loudest_section 'long_thanks/*.wav' thanks/
# >  /tmp/extract_loudest_section/gen/bin/extract_loudest_section 'long_hello/*.wav' hello/
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt update \
    && apt install --yes --no-install-recommends \
        unzip \
        wget \
        software-properties-common \
        build-essential \
        git
RUN git clone https://github.com/petewarden/extract_loudest_section
RUN make -C extract_loudest_section

RUN addgroup --gid 1000 mark
RUN adduser --disabled-password --gecos '' --uid 1000 --gid 1000 mark
USER mark
WORKDIR /home/mark/
ENV PATH /home/mark/.local/bin:$PATH