NVCC ?= nvcc
ARCH ?= -arch=sm_90
NVCCFLAGS ?= -O2 -lineinfo

all: clone_semantics

clone_semantics: clone_semantics.cu
	$(NVCC) $(NVCCFLAGS) $(ARCH) $< -o $@

clone_external: clone_external.cu
	$(NVCC) $(NVCCFLAGS) $(ARCH) $< -o $@

clean:
	rm -f clone_semantics clone_external

.PHONY: all clean
