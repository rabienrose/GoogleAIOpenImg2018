import functools
import json
import os
import tensorflow as tf

from builders import dataset_builder
from builders import model_builder
from utils import config_util
from utils import dataset_util
from builders import optimizer_builder
from builders import preprocessor_builder
from core import batcher
from core import preprocessor
from core import standard_fields as fields
from utils import ops as util_ops
from utils import variables_helper
slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)

pipeline_addr = '/home/yiming/Documents/data/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28/pipeline.config'
output_addr ='/home/yiming/Documents/data/output'
batch_size=4

def main(_):
    configs = config_util.get_configs_from_pipeline_file(pipeline_addr)
    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']
    data_augmentation_options = [preprocessor_builder.build(step) for step in train_config.data_augmentation_options]
    global_step = slim.create_global_step()
    clones = []
    for i in range(0, batch_size):
        with tf.variable_scope(tf.get_variable_scope(), reuse=True if i > 0 else None):
            tensor_dict = dataset_util.make_initializable_iterator(dataset_builder.build(input_config)).get_next()
            tensor_dict[fields.InputDataFields.image] = tf.expand_dims(tensor_dict[fields.InputDataFields.image], 0)
            images = tensor_dict[fields.InputDataFields.image]
            float_images = tf.to_float(images)
            tensor_dict[fields.InputDataFields.image] = float_images
            tensor_dict = preprocessor.preprocess(tensor_dict, data_augmentation_options)
            detection_model = model_builder.build(model_config=model_config, is_training=True)
            image = tensor_dict[fields.InputDataFields.image]
            location_gt = tensor_dict[fields.InputDataFields.groundtruth_boxes]
            classes_gt = tf.cast(tensor_dict[fields.InputDataFields.groundtruth_classes], tf.int32)
            classes_gt -= 1
            classes_gt = util_ops.padded_one_hot_encoding(indices=classes_gt, depth=model_config.faster_rcnn.num_classes, left_pad=0)
            image, true_image_shapes = detection_model.preprocess(image)
            detection_model.provide_groundtruth(location_gt, classes_gt)
            prediction_dict = detection_model.predict(image, true_image_shapes)
            losses_dict = detection_model.loss(prediction_dict, true_image_shapes)
            for loss_tensor in losses_dict.values():
                tf.losses.add_loss(loss_tensor)
            clones.append(losses_dict)
    grads_and_vars = []
    clones_losses = []
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    for clone in clones:
        all_losses = []
        clone_losses = tf.get_collection(tf.GraphKeys.LOSSES, clone.scope)
        clone_loss = tf.add_n(clone_losses, name='clone_loss')
        if batch_size > 1:
            clone_loss = tf.div(clone_loss, 1.0 * batch_size, name='scaled_clone_loss')
        all_losses.append(clone_loss)
        if regularization_losses==None:
            regularization_loss = tf.add_n(regularization_losses, name='regularization_loss')
            all_losses.append(regularization_loss)
        sum_loss = tf.add_n(all_losses)
        tf.summary.scalar('Losses/regularization_loss', regularization_loss)
        clones_losses.append(sum_loss)
        regularization_losses = None
    total_loss = tf.add_n(clones_losses, name='total_loss')
    clone_grad = optimizer.compute_gradients(total_loss)
    sum_grads = []
    for grad_and_vars in zip(*clone_grad):
        grads = []
        var = grad_and_vars[0][1]
        for g, v in grad_and_vars:
            assert v == var
            if g is not None:
                grads.append(g)
        if grads:
            if len(grads) > 1:
                sum_grad = tf.add_n(grads, name=var.op.name + '/sum_grads')
            else:
                sum_grad = grads[0]
            sum_grads.append((sum_grad, var))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    training_optimizer, optimizer_summary_vars = optimizer_builder.build(train_config.optimizer)
    var_map = detection_model.restore_map(fine_tune_checkpoint_type='detection',load_all_detection_checkpoint_vars=(train_config.load_all_detection_checkpoint_vars))
    available_var_map = (variables_helper.get_variables_available_in_checkpoint(var_map, train_config.fine_tune_checkpoint))
    init_saver = tf.train.Saver(available_var_map)

    if train_config.freeze_variables:
        grads_and_vars = variables_helper.freeze_gradients_matching_regex(grads_and_vars, train_config.freeze_variables)
    if train_config.gradient_clipping_by_norm > 0:
        with tf.name_scope('clip_grads'):
            grads_and_vars = slim.learning.clip_gradient_norms(grads_and_vars, train_config.gradient_clipping_by_norm)
    grad_updates = training_optimizer.apply_gradients(grads_and_vars,global_step=global_step)
    update_ops.append(grad_updates)
    update_op = tf.group(*update_ops, name='update_barrier')
    with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')

    global_summaries=[]
    global_summaries.append(tf.summary.scalar('Losses/' + loss_tensor.op.name, loss_tensor))
    global_summaries.append(tf.summary.scalar('Losses/TotalLoss', tf.losses.get_total_loss()))
    summary_op = tf.summary.merge(global_summaries, name='summary_op')
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        init_saver.restore(sess, train_config.fine_tune_checkpoint)
        writer = tf.summary.FileWriter("logs/", sess.graph)
        i = -1
        while True:
            before_time = time.perf_counter()
            i = i + 1
            sess.run(train_step)
            after_time = time.perf_counter()
            step_time = after_time - before_time
            print(step_time)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
