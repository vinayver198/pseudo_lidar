import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8  # one cluster for each type
NUM_OBJECT_POINT = 512
g_type2class = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3,
                'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7}
g_class2type = {g_type2class[t]: t for t in g_type2class}
g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
g_type_mean_size = {'Car': np.array([3.88311640418, 1.62856739989, 1.52563191462]),
                    'Van': np.array([5.06763659, 1.9007158, 2.20532825]),
                    'Truck': np.array([10.13586957, 2.58549199, 3.2520595]),
                    'Pedestrian': np.array([0.84422524, 0.66068622, 1.76255119]),
                    'Person_sitting': np.array([0.80057803, 0.5983815, 1.27450867]),
                    'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127]),
                    'Tram': np.array([16.17150617, 2.53246914, 3.53079012]),
                    'Misc': np.array([3.64300781, 1.54298177, 1.92320313])}

g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clusters
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i, :] = g_type_mean_size[g_class2type[i]]


@tf.function
def tf_gather_object_pc(point_cloud, mask, npoints=512):
    ''' Gather object point clouds according to predicted masks.
    Input:
        point_cloud: TF tensor in shape (B,N,C)
        mask: TF tensor in shape (B,N) of 0 (not pick) or 1 (pick)
        npoints: int scalar, maximum number of points to keep (default: 512)
    Output:
        object_pc: TF tensor in shape (B,npoint,C)
        indices: TF int tensor in shape (B,npoint,2)
    '''

    def mask_to_indices(mask):
        indices = np.zeros((mask.shape[0], npoints, 2), dtype=np.int32)
        for i in range(mask.shape[0]):
            pos_indices = np.where(mask[i, :] > 0.5)[0]
            # skip cases when pos_indices is empty
            if len(pos_indices) > 0:
                if len(pos_indices) > npoints:
                    choice = np.random.choice(len(pos_indices),
                                              npoints, replace=False)
                else:
                    choice = np.random.choice(len(pos_indices),
                                              npoints - len(pos_indices), replace=True)
                    choice = np.concatenate((np.arange(len(pos_indices)), choice))
                np.random.shuffle(choice)
                indices[i, :, 1] = pos_indices[choice]
            indices[i, :, 0] = i
        return indices

    indices = tf.py_function(mask_to_indices, [mask], tf.int32)
    object_pc = tf.gather_nd(point_cloud, indices)
    return object_pc, indices


@tf.function
def mask_point_cloud(point_cloud, logits, xyz_only=True):
    batch_size = point_cloud.get_shape()[0]
    num_point = point_cloud.get_shape()[1]
    mask = tf.slice(logits, [0, 0, 0], [-1, -1, 1]) < tf.slice(logits, [0, 0, 1], [-1, -1, 1])
    mask = tf.cast(mask, dtype=tf.float32)

    mask_count = tf.tile(tf.reduce_sum(mask, axis=1, keepdims=True), [1, 1, 3])

    point_cloud_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])
    mask_xyz_mean = tf.reduce_sum(tf.tile(mask, [1, 1, 3]) * point_cloud_xyz, axis=1, keepdims=True)
    mask = tf.squeeze(mask, axis=[2])
    mask_xyz_mean = mask_xyz_mean / tf.maximum(mask_count, 1)

    point_cloud_xyz_stage_1 = point_cloud_xyz - tf.tile(mask_xyz_mean, [1, num_point, 1])

    if xyz_only:
        point_cloud_stage_1 = point_cloud_xyz_stage_1
    else:
        point_cloud_features = tf.slice(point_cloud, [0, 0, 3], [-1, -1, -1])
        point_cloud_stage_1 = tf.concat([point_cloud_xyz_stage_1, point_cloud_features], axis=-1)
    num_channels = point_cloud_stage_1.get_shape()[2]
    object_point_cloud, _ = tf_gather_object_pc(point_cloud_stage_1, mask, NUM_OBJECT_POINT)
    object_point_cloud.set_shape([batch_size, NUM_OBJECT_POINT, num_channels])
    return object_point_cloud, tf.squeeze(mask_xyz_mean, axis=1)


@tf.function
def parse_output_to_tensors(output, end_points):
    ''' Parse batch output to separate tensors (added to end_points)
    Input:
        output: TF tensor in shape (B,3+2*NUM_HEADING_BIN+4*NUM_SIZE_CLUSTER)
        end_points: dict
    Output:
        end_points: dict (updated)
    '''

    center = tf.slice(output, [0, 0], [-1, 3])
    end_points['center_boxnet'] = center

    heading_scores = tf.slice(output, [0, 3], [-1, NUM_HEADING_BIN])
    heading_residuals_normalized = tf.slice(output, [0, 3 + NUM_HEADING_BIN],
                                            [-1, NUM_HEADING_BIN])
    end_points['heading_scores'] = heading_scores  # BxNUM_HEADING_BIN
    end_points['heading_residuals_normalized'] = \
        heading_residuals_normalized  # BxNUM_HEADING_BIN (-1 to 1)
    end_points['heading_residuals'] = \
        heading_residuals_normalized * (np.pi / NUM_HEADING_BIN)  # BxNUM_HEADING_BIN

    size_scores = tf.slice(output, [0, 3 + NUM_HEADING_BIN * 2],
                           [-1, NUM_SIZE_CLUSTER])  # BxNUM_SIZE_CLUSTER
    size_residuals_normalized = tf.slice(output,
                                         [0, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER], [-1, NUM_SIZE_CLUSTER * 3])

    size_residuals_normalized = tf.reshape(size_residuals_normalized,[-1, NUM_SIZE_CLUSTER, 3])  # BxNUM_SIZE_CLUSTERx3

    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['size_residuals'] = size_residuals_normalized * \
                                   tf.expand_dims(tf.constant(g_mean_size_arr, dtype=tf.float32), 0)

    return end_points
