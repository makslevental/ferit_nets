#!/usr/bin/env bash
mpirun -np 4 -H localhost:1,horovod0:1,horovod1:1,horovod2:1 \
   -bind-to none -map-by slot \
   -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
   -x NCCL_SOCKET_IFNAME=^lo,docker0 \
   -mca pml ob1 -mca btl ^openib \
   -mca btl_tcp_if_exclude lo,docker0 \
   python /examples/pytorch_mnist.py

time mpirun -np 4 -H localhost:1,horovod0:1,horovod1:1,horovod2:1 \
   -bind-to none -map-by slot \
   -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
   -x NCCL_SOCKET_IFNAME=^lo,docker0 \
   -mca pml ob1 -mca btl ^openib \
   -mca btl_tcp_if_exclude lo,docker0 \
   python /examples/pytorch_mnist.py

