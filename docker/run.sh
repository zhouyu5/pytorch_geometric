# docker build ./  \
#     -f docker/Dockerfile.pvc -t xpu_pyg:v1 \
#     --build-arg http_proxy=${http_proxy} \
#     --build-arg https_proxy=${https_proxy} 


docker run \
    --name="pvc_train" \
    --privileged \
    -w /workspace/  \
    -v /dev/dri:/dev/dri \
    -v $(pwd):/work \
    --ipc=host \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    -e no_proxy=$no_proxy \
    -itd nathanzz2/xpu_pyg:v1


docker exec -it "pvc_train" bash
# docker stop pvc_train && docker rm pvc_train


