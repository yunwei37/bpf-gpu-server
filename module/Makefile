# SPDX-License-Identifier: Apache-2.0
#
# bpf-gpu-server kernel TCP TX module. Builds against the running kernel's headers.
# The Ubuntu 24.04 default gcc-13 lacks -fmin-function-alignment=16 (this
# kernel's arch flags), so use gcc-15 from ppa:ubuntu-toolchain-r/test.
obj-m := bpf_gpu_server_tx.o
CC ?= gcc-15

KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) CC=$(CC) -C $(KDIR) M=$(CURDIR) modules

clean:
	$(MAKE) CC=$(CC) -C $(KDIR) M=$(CURDIR) clean

.PHONY: all clean
