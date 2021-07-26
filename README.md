# 3d detection on Pseudo Lidar using Frustum PointNet

## The work is divided in 2 phases : <br>
1. Generate pseudo lidar
2. Training Frustum PointNet on pseudo lidar data

## Generate Pseudo Lidar <br>
To generate data [Monodepth2](https://arxiv.org/abs/1806.01260) is used. I have used monodepth implementation using [mxnet](https://cv.gluon.ai/build/examples_depth/index.html) 