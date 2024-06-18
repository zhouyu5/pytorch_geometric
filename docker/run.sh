docker build docker/  \
    -f docker/Dockerfile.cpu -t cpu_pyg:v1 \
    --build-arg http_proxy=${http_proxy} \
    --build-arg https_proxy=${https_proxy} 


docker run \
    --name="pvc_train" \
    --privileged \
    -w /workspace/  \
    -v /dev/dri:/dev/dri \
    -v $(pwd):/work \
    -p 2222:2222 \
    -p 11111:11111 \
    -p 11112:11112 \
    -p 11113:11113 \
    -v /etc/hosts:/etc/hosts \
    --ipc=host \
    --network=host \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    -e no_proxy=$no_proxy \
    -itd nathanzz2/cpu_pyg:v1 \
&& docker exec -it "pvc_train" bash

docker stop pvc_train && docker rm pvc_train
docker stop pvc_train && docker rm pvc_train && docker system prune -a -f


