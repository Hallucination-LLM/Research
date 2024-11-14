# Research

```
docker build -t hallu:1.1 -f Dockerfile.local_gemma .
```

```
docker run --name hallu_app2 -v ./api/:/app/ -d -p 8000:8000 --gpus "device=0" hallu:1.1
```

