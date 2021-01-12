# coding=utf-8

import os, shutil
from tensorflow.python.framework import graph_util

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# ------- ablert
import rc

tf = rc.tf

checkpoint_path = 'outputs/albert_batch64_max512_lr2e-05_F1_82.000/best_model.weights'
rc.model.load_weights(checkpoint_path)

max_seq_length = 512

config = tf.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    # 输出 pb
    with tf.gfile.FastGFile('model.pb', 'wb') as f:
        graph_def = sess.graph.as_graph_def()
        output_nodes = ['permute/transpose']
        print('outputs:', output_nodes)
        #print('\n'.join([n.name for n in tf.get_default_graph().as_graph_def().node])) # 所有层的名字
        output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, output_nodes)
        f.write(output_graph_def.SerializeToString())

    # save_model 输出 , for goland 测试
    if os.path.exists('outputs/saved-model'):
        shutil.rmtree("outputs/saved-model") 
    builder = tf.saved_model.builder.SavedModelBuilder("outputs/saved-model")
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING], clear_devices=True)
    builder.save()  
