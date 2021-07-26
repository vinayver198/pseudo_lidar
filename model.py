import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, GlobalAvgPool2D, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from model_utils import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, mask_point_cloud,parse_output_to_tensors
from losses import get_loss

tf.compat.v1.disable_eager_execution()

class CenterRegressionNet(tf.keras.layers.Layer):
    def __init__(self, num_point=2048):
        super(CenterRegressionNet, self).__init__()
        self.conv1 = Conv2D(filters=128, kernel_size=(1, 1),
                            padding='VALID', activation='relu')
        self.conv2 = Conv2D(filters=128, kernel_size=(1, 1),
                            padding='VALID', activation='relu')
        self.conv3 = Conv2D(filters=256, kernel_size=(1, 1),
                            padding='VALID', activation='relu')
        self.maxpool_1 = MaxPool2D([num_point, 1], padding='VALID')
        self.dense1 = Dense(units=256, activation='relu')
        self.dense2 = Dense(units=128, activation='relu')
        self.outputs = Dense(3, activation=None)

    def call(self, inputs, one_hot_vec):
        num_point = inputs.get_shape()[1]
        x = tf.expand_dims(inputs, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool_1(x)
        x = tf.squeeze(x, axis=[1, 2])
        x = tf.concat([x, one_hot_vec], axis=1)
        x = self.dense1(x)
        x = self.dense2(x)
        outputs = self.outputs(x)
        return outputs


class BoxEstimationNet(tf.keras.layers.Layer):
    def __init__(self, num_point=2048):
        super(BoxEstimationNet, self).__init__()
        self.conv1 = Conv2D(filters=128, kernel_size=(1, 1),
                            padding='VALID', activation='relu')
        self.conv2 = Conv2D(filters=128, kernel_size=(1, 1),
                            padding='VALID', activation='relu')
        self.conv3 = Conv2D(filters=256, kernel_size=(1, 1),
                            padding='VALID', activation='relu')
        self.conv4 = Conv2D(filters=512, kernel_size=(1, 1),
                            padding='VALID', activation='relu')
        self.maxpool_1 = MaxPool2D([num_point, 1], padding='VALID')
        self.dense1 = Dense(units=512, activation='relu')
        self.dense2 = Dense(units=256, activation='relu')
        self.outputs = Dense(3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4,
                             activation=None)

    def call(self, inputs, one_hot_vec):
        num_point = inputs.get_shape()[1]
        x = tf.expand_dims(inputs, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool_1(x)
        x = tf.squeeze(x, axis=[1, 2])
        x = tf.concat([x, one_hot_vec], axis=1)
        x = self.dense1(x)
        x = self.dense2(x)
        outputs = self.outputs(x)
        return outputs


class SegmentationNet(tf.keras.layers.Layer):
    def __init__(self, num_point=2048):
        super(SegmentationNet, self).__init__()
        self.conv1 = Conv2D(filters=64, kernel_size=(1, 1),
                            padding='VALID', activation='relu')
        self.conv2 = Conv2D(filters=64, kernel_size=(1, 1),
                            padding='VALID', activation='relu')
        self.point_features = Conv2D(filters=64, kernel_size=(1, 1),
                                     padding='VALID', activation='relu')
        self.conv3 = Conv2D(filters=128, kernel_size=(1, 1),
                            padding='VALID', activation='relu')
        self.conv4 = Conv2D(filters=1024, kernel_size=(1, 1),
                            padding='VALID', activation='relu')
        self.global_features = MaxPool2D((num_point, 1), padding='VALID')
        self.conv5 = Conv2D(filters=512, kernel_size=(1, 1),
                            padding='VALID', activation='relu')
        self.conv6 = Conv2D(filters=256, kernel_size=(1, 1),
                            padding='VALID', activation='relu')
        self.conv7 = Conv2D(filters=128, kernel_size=(1, 1),
                            padding='VALID', activation='relu')
        self.conv8 = Conv2D(filters=128, kernel_size=(1, 1),
                            padding='VALID', activation='relu')
        self.dropout = Dropout(rate=0.7)
        self.outputs = Conv2D(2, (1, 1), padding='VALID',
                              activation=None)

    def call(self, inputs, one_hot_vec):
        num_point = inputs.get_shape()[1]
        x = tf.expand_dims(inputs, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        point_features = self.point_features(x)
        x = self.conv3(point_features)
        x = self.conv4(x)
        global_features = self.global_features(x)
        global_features = tf.concat([global_features,
                                     tf.expand_dims(tf.expand_dims(
                                         one_hot_vec, 1), 1)], axis=3)
        global_features_expand = tf.tile(global_features, [1, num_point, 1, 1])
        concatenate_features = tf.concat(axis=3, values=[point_features, global_features_expand])
        x = self.conv5(concatenate_features)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.dropout(x)
        logits = self.outputs(x)
        logits = tf.squeeze(logits, [2])
        return logits

def get_model(inputs,one_hot_vec):
    end_points = dict()
    segment_layer = SegmentationNet(inputs.get_shape()[1])
    seg_outputs = segment_layer(inputs, one_hot_vec)
    end_points['mask_logits'] = seg_outputs
    object_point_cloud, mask_xyz_mean = mask_point_cloud(inputs, seg_outputs)

    center_regression_layer = CenterRegressionNet(object_point_cloud.get_shape()[1])
    center_delta = center_regression_layer(object_point_cloud, one_hot_vec)

    stage1_center = center_delta + mask_xyz_mean
    end_points['stage1_center'] = stage1_center
    object_point_cloud_xyz_new = object_point_cloud - tf.expand_dims(center_delta, 1)

    box_estimation_layer = BoxEstimationNet(object_point_cloud_xyz_new.get_shape()[1])

    box_outputs = box_estimation_layer(object_point_cloud_xyz_new, one_hot_vec)
    print(f'Keys : {end_points.keys()}')
    end_points = parse_output_to_tensors(box_outputs, end_points)
    end_points['center'] = end_points['center_boxnet'] + stage1_center
    print(f'Keys : {end_points.keys()}')
    outputs = [end_points['mask_logits'],end_points['stage1_center'],end_points['center_boxnet'],
               end_points['heading_scores'],end_points['heading_residuals_normalized'],
               end_points['heading_residuals'],end_points['size_scores'],end_points['size_residuals_normalized'],
               end_points['size_residuals'],end_points['center']]
    model = Model(inputs=[inputs, one_hot_vec], outputs=outputs)
    return model,end_points


if __name__=='__main__':
    inputs = Input(shape=(2048, 3))
    one_hot_vec = Input(shape=(3))

    model, end_points = get_model(inputs, one_hot_vec)

    outputs = model.predict([tf.zeros((32, 2048, 3)), tf.zeros((32, 3))], steps=1)
    for output in outputs:
        print(output.shape)
    loss = get_loss(tf.zeros((32, 2048), dtype=tf.int32), tf.zeros((32, 3)), tf.zeros((32,), dtype=tf.int32),
                    tf.zeros((32,)), tf.zeros((32,), dtype=tf.int32), tf.zeros((32, 3)), outputs)