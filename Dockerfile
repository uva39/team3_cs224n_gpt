FROM continuumio/miniconda3:latest

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    git \
    vim \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN echo "root:root" | chpasswd && \
    mkdir -p /run/sshd && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

WORKDIR /workspace

COPY env.yml /tmp/env.yml

RUN conda env create -f /tmp/env.yml && \
    conda clean -afy && \
    rm -rf /root/.cache/pip /tmp/* /var/tmp/*

ENV CONDA_DEFAULT_ENV=cs224n_dfp
ENV PATH=/opt/conda/envs/cs224n_dfp/bin:/opt/conda/bin:$PATH

RUN python --version && \
    pip --version && \
    python -c "import torch; import transformers; print(torch.__version__); print(transformers.__version__)"

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]