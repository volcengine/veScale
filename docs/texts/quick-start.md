# Quick Start

## Install veScale

Make sure we are in root of the veScale repo.

```bash
pip3 install -r requirements.txt && pip3 install -e .
```

This will install **veScale** and its dependencies.

## Build Docker Image

Make sure we are in root of the veScale repo.

```bash
docker build .
```

Once the building process is finished, you can `docker run` with the id.

## Run Test Code

Make sure we are in root of the veScale repo.

```bash
./scripts/run_test.sh
```
