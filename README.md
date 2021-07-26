# 3d detection on Pseudo Lidar using Frustum PointNet

## The work is divided in 2 phases : <br>
1. Generate pseudo lidar
2. Training Frustum PointNet on pseudo lidar data

## Generate Pseudo Lidar <br>
For generating pseudo lidar data [Monodepth2](https://arxiv.org/abs/1806.01260) is used. I have used monodepth implementation using [mxnet](https://cv.gluon.ai/build/examples_depth/index.html) .

To generate data first download the kitti 3d object detection dataset from [KITTI's](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) official website . <br>
Once downloaded arrange the dataset in below format .
```angular2html
KITTI/object/
    
    train.txt
    val.txt
    test.txt 
    
    training/
        calib/
        image_2/ #left image
        image_3/ #right image
        label_2/
        velodyne/ 

    testing/
        calib/
        image_2/
        image_3/
        velodyne/
```

Run the following command :

```

python generate_disp.py --root_dir Path of kitti dataset

``` 


This generates a disparity folder both in training and testing folder and stores predicted images in that.

To generate the pseudo point cloud run : 

```

python generate_lidar.py --calib_dir <kitti calib path> --disparity_dir <Disparity path> --save_dir <~/kitti_data/training/pseudo-lidar_velodyne>

```

This generates a folder for pseudo velodyne and stores data in .bin format

## Train Frustum PointNet Model 

1. To generate frustum data : 

```

cd kitti/
python prepare_data.py --gen_train --gen_val --gen_val_rgb_detection

```

2. To train the model 

```
python train.py 

```

