TEST_PATH=./
DOCKER_IMAGE=poldracklab/fmriprep
VERSION := $(shell python version.py)
$( eval VERSION := $( shell python version.py ))

.PHONY: clean-pyc
clean-pyc:
		find . -name '__pycache__' -type d -exec rm -fr {} +
		find . -name '*.pyc' -exec rm --force {} +
		find . -name '*.pyo' -exec rm --force {} +
		find . -name '*~' -exec rm --force  {} +

.PHONY: clean-build
clean-build:
		rm --force --recursive build/
		rm --force --recursive dist/
		rm --force --recursive *.egg-info
		rm --force --recursive src/

.PHONY: tag
tag:
		git tag -a $(VERSION) -m "Version ${VERSION}"
		git push origin $(VERSION)
		git push upstream $(VERSION)

lint:
		pylint ./fmriprep/

.PHONY: test
test: clean-pyc
		docker build -t $(DOCKER_IMAGE):latest .
		docker run -ti --rm --entrypoint="/usr/bin/run_unittests" -v $PWD:/root/src/fmriprep -v ~/.cache/:/root/.cache/ $(DOCKER_IMAGE):latest

dist: clean-build clean-pyc
		python setup.py sdist

.PHONY: docker-build
docker-build:
		docker build \
			-f ./docker/Dockerfile_py27 \
			-t $(DOCKER_IMAGE):$(VERSION)-python27 -t $(DOCKER_IMAGE):latest -t $(DOCKER_IMAGE):$(VERSION) .
		docker build \
			-f ./docker/Dockerfile_py35 \
			-t $(DOCKER_IMAGE):$(VERSION)-python35 .

.PHONY: docker
docker: docker-build
		docker push $(DOCKER_IMAGE):$(VERSION)-python27
		docker push $(DOCKER_IMAGE):$(VERSION)
		docker push $(DOCKER_IMAGE):latest
		docker push $(DOCKER_IMAGE):$(VERSION)-python35

.PHONY: release
release: clean-build tag docker
		python setup.py sdist
		twine upload dist/*

singularity: docker
	mkdir -p build/singularity
	docker run --privileged -ti --rm  \
    	-v /var/run/docker.sock:/var/run/docker.sock \
    	-v $(shell pwd)/build/singularity:/output \
    	singularityware/docker2singularity \
    	$(DOCKER_IMAGE):$(VERSION)

