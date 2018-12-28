#!/usr/bin/env bash
nvidia-docker run -it --network=host horovod:latest \
    bash -c "/usr/sbin/sshd -p 12345; watch -n 0.1 nvidia-smi"

