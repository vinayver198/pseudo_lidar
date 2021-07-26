import tensorflow as tf
import numpy as np
from model_utils import NUM_HEADING_BIN,NUM_SIZE_CLUSTER,g_mean_size_arr


def get_box3d_corners_helper(centers, headings, sizes):
    """ TF layer. Input: (N,3), (N,), (N,3), Output: (N,8,3) """
    # print '-----', centers
    N = centers.shape[0]
    l = tf.slice(sizes, [0, 0], [-1, 1])  # (N,1)
    w = tf.slice(sizes, [0, 1], [-1, 1])  # (N,1)
    h = tf.slice(sizes, [0, 2], [-1, 1])  # (N,1)
    # print l,w,h
    x_corners = tf.concat([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], axis=1)  # (N,8)
    y_corners = tf.concat([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], axis=1)  # (N,8)
    z_corners = tf.concat([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], axis=1)  # (N,8)
    corners = tf.concat([tf.expand_dims(x_corners, 1), tf.expand_dims(y_corners, 1), tf.expand_dims(z_corners, 1)],
                        axis=1)  # (N,3,8)
    # print x_corners, y_corners, z_corners
    c = tf.cos(headings)
    s = tf.sin(headings)
    ones = tf.ones([N], dtype=tf.float32)
    zeros = tf.zeros([N], dtype=tf.float32)
    row1 = tf.stack([c, zeros, s], axis=1)  # (N,3)
    row2 = tf.stack([zeros, ones, zeros], axis=1)
    row3 = tf.stack([-s, zeros, c], axis=1)
    R = tf.concat([tf.expand_dims(row1, 1), tf.expand_dims(row2, 1), tf.expand_dims(row3, 1)], axis=1)  # (N,3,3)
    # print row1, row2, row3, R, N
    corners_3d = tf.matmul(R, corners)  # (N,3,8)
    corners_3d += tf.tile(tf.expand_dims(centers, 2), [1, 1, 8])  # (N,3,8)
    corners_3d = tf.transpose(corners_3d, perm=[0, 2, 1])  # (N,8,3)
    return corners_3d


def get_box3d_corners(center, heading_residuals, size_residuals):
    """ TF layer.
    Inputs:
        center: (B,3)
        heading_residuals: (B,NH)
        size_residuals: (B,NS,3)
    Outputs:
        box3d_corners: (B,NH,NS,8,3) tensor
    """
    batch_size = center.shape[0]
    heading_bin_centers = tf.constant(np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN), dtype=tf.float32)  # (NH,)
    headings = heading_residuals + tf.expand_dims(heading_bin_centers, 0)  # (B,NH)

    mean_sizes = tf.expand_dims(tf.constant(g_mean_size_arr, dtype=tf.float32), 0) + size_residuals  # (B,NS,1)
    sizes = mean_sizes + size_residuals  # (B,NS,3)
    sizes = tf.tile(tf.expand_dims(sizes, 1), [1, NUM_HEADING_BIN, 1, 1])  # (B,NH,NS,3)
    headings = tf.tile(tf.expand_dims(headings, -1), [1, 1, NUM_SIZE_CLUSTER])  # (B,NH,NS)
    centers = tf.tile(tf.expand_dims(tf.expand_dims(center, 1), 1),
                      [1, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1])  # (B,NH,NS,3)

    N = batch_size * NUM_HEADING_BIN * NUM_SIZE_CLUSTER
    corners_3d = get_box3d_corners_helper(tf.reshape(centers, [N, 3]), tf.reshape(headings, [N]),
                                          tf.reshape(sizes, [N, 3]))

    return tf.reshape(corners_3d, [batch_size, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3])

def huber_loss(error, delta):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return tf.reduce_mean(losses)

def get_loss(mask_label, center_label, \
             heading_class_label, heading_residual_label, \
             size_class_label, size_residual_label, \
             end_points, \
             corner_loss_weight=10.0, \
             box_loss_weight=1.0):
    ''' Loss functions for 3D object detection.
    Input:
        mask_label: TF int32 tensor in shape (B,N)
        center_label: TF tensor in shape (B,3)
        heading_class_label: TF int32 tensor in shape (B,)
        heading_residual_label: TF tensor in shape (B,)
        size_class_label: TF tensor int32 in shape (B,)
        size_residual_label: TF tensor tensor in shape (B,)
        end_points: dict, outputs from our model
        corner_loss_weight: float scalar
        box_loss_weight: float scalar
    Output:
        total_loss: TF scalar tensor
            the total_loss is also added to the losses collection
    '''
    # 3D Segmentation loss
    print(end_points[0])
    mask_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_points[0], labels=mask_label))
    print(mask_loss)
    tf.summary.scalar('3d mask loss', mask_loss)

    # Center regression losses
    center_dist = tf.norm(center_label - end_points[-1], axis=-1)
    center_loss = huber_loss(center_dist, delta=2.0)
    tf.summary.scalar('center loss', center_loss)
    stage1_center_dist = tf.norm(center_label - \
        end_points[1], axis=-1)
    stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)
    tf.summary.scalar('stage1 center loss', stage1_center_loss)

    # Heading loss
    heading_class_loss = tf.reduce_mean( \
        tf.nn.sparse_softmax_cross_entropy_with_logits( \
        logits=end_points[3], labels=heading_class_label))
    tf.summary.scalar('heading class loss', heading_class_loss)

    hcls_onehot = tf.one_hot(heading_class_label,
        depth=NUM_HEADING_BIN,
        on_value=1, off_value=0, axis=-1) # BxNUM_HEADING_BIN
    heading_residual_normalized_label = \
        heading_residual_label / (np.pi/NUM_HEADING_BIN)
    heading_residual_normalized_loss = huber_loss(tf.reduce_sum( \
        end_points[4]*tf.cast(hcls_onehot,dtype=tf.float32), axis=1) - \
        heading_residual_normalized_label, delta=1.0)
    tf.summary.scalar('heading residual normalized loss',
        heading_residual_normalized_loss)

    # Size loss
    size_class_loss = tf.reduce_mean( \
        tf.nn.sparse_softmax_cross_entropy_with_logits( \
        logits=end_points[6], labels=size_class_label))
    tf.summary.scalar('size class loss', size_class_loss)

    scls_onehot = tf.one_hot(size_class_label,
        depth=NUM_SIZE_CLUSTER,
        on_value=1, off_value=0, axis=-1) # BxNUM_SIZE_CLUSTER
    scls_onehot_tiled = tf.tile(tf.expand_dims( \
        tf.cast(scls_onehot,dtype=tf.float32), -1), [1,1,3]) # BxNUM_SIZE_CLUSTERx3
    predicted_size_residual_normalized = tf.reduce_sum( \
        end_points[7]*scls_onehot_tiled, axis=[1]) # Bx3

    mean_size_arr_expand = tf.expand_dims( \
        tf.constant(g_mean_size_arr, dtype=tf.float32),0) # 1xNUM_SIZE_CLUSTERx3
    mean_size_label = tf.reduce_sum( \
        scls_onehot_tiled * mean_size_arr_expand, axis=[1]) # Bx3
    size_residual_label_normalized = size_residual_label / mean_size_label
    size_normalized_dist = tf.norm( \
        size_residual_label_normalized - predicted_size_residual_normalized,
        axis=-1)
    size_residual_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)
    tf.summary.scalar('size residual normalized loss',
        size_residual_normalized_loss)

    # Corner loss
    # We select the predicted corners corresponding to the
    # GT heading bin and size cluster.
    corners_3d = get_box3d_corners(end_points[-1],
        end_points[5],
        end_points[8]) # (B,NH,NS,8,3)
    gt_mask = tf.tile(tf.expand_dims(hcls_onehot, 2), [1,1,NUM_SIZE_CLUSTER]) * \
        tf.tile(tf.expand_dims(scls_onehot,1), [1,NUM_HEADING_BIN,1]) # (B,NH,NS)
    corners_3d_pred = tf.reduce_sum( \
        tf.cast(tf.expand_dims(tf.expand_dims(gt_mask,-1),-1),dtype=tf.float32) * corners_3d,
        axis=[1,2]) # (B,8,3)

    heading_bin_centers = tf.constant( \
        np.arange(0,2*np.pi,2*np.pi/NUM_HEADING_BIN), dtype=tf.float32) # (NH,)
    heading_label = tf.expand_dims(heading_residual_label,1) + \
        tf.expand_dims(heading_bin_centers, 0) # (B,NH)
    heading_label = tf.reduce_sum(tf.cast(hcls_onehot,dtype=tf.float32)*heading_label, 1)
    mean_sizes = tf.expand_dims( \
        tf.constant(g_mean_size_arr, dtype=tf.float32), 0) # (1,NS,3)
    size_label = mean_sizes + \
        tf.expand_dims(size_residual_label, 1) # (1,NS,3) + (B,1,3) = (B,NS,3)
    size_label = tf.reduce_sum( \
        tf.expand_dims(tf.cast(scls_onehot,dtype=tf.float32),-1)*size_label, axis=[1]) # (B,3)
    corners_3d_gt = get_box3d_corners_helper( \
        center_label, heading_label, size_label) # (B,8,3)
    corners_3d_gt_flip = get_box3d_corners_helper( \
        center_label, heading_label+np.pi, size_label) # (B,8,3)

    corners_dist = tf.minimum(tf.norm(corners_3d_pred - corners_3d_gt, axis=-1),
        tf.norm(corners_3d_pred - corners_3d_gt_flip, axis=-1))
    corners_loss = huber_loss(corners_dist, delta=1.0)
    tf.summary.scalar('corners loss', corners_loss)

    # Weighted sum of all losses
    total_loss = mask_loss + box_loss_weight * (center_loss + \
        heading_class_loss + size_class_loss + \
        heading_residual_normalized_loss*20 + \
        size_residual_normalized_loss*20 + \
        stage1_center_loss + \
        corner_loss_weight*corners_loss)
    tf.summary.scalar('losses', total_loss)

    return total_loss