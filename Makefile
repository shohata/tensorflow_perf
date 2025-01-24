DOCKER := docker
IMAGE_TAG := v2.14.0
IMAGE_NAME := tmc/tensorflow_cc
CONTINER_NAME := gpu_perf
DATA_DIR := /data
MODEL_DIR := /saved_model
CMD := bash

.PHONY: build build-devel run

build:
	$(DOCKER) build --network host -t $(IMAGE_NAME):$(IMAGE_TAG) docker/

build-devel:
	$(DOCKER) build --network host -t $(IMAGE_NAME):devel --target tensorflow-builder docker/

run:
	$(DOCKER) run --rm -it \
		--network host \
		--gpus all \
		-w /workspaces \
		--name $(CONTINER_NAME) \
		-u $(shell id -u):$(shell id -g) \
		-v $(shell pwd):/workspaces \
		-v $(DATA_DIR):/data \
		-v $(MODEL_DIR):/saved_model \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		$(CMD)
