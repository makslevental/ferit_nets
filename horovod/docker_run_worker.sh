#!/usr/bin/env bash
nvidia-docker run -it -v /export/lv_data/landmines6/ferit_data/ferit_nets:/ferit_nets --network=host horovod:latest \
    bash -c "/usr/sbin/sshd -p 12345; watch -n 0.1 nvidia-smi"

