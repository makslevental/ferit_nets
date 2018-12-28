#!/usr/bin/env bash
nvidia-docker run -it --network=host horovod:latest \
    bash -c "bash /examples/mpirun.sh"


