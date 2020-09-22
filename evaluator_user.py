import os, sys, cv2, random, time, copy
import tensorflow as tf
import numpy as np
from glob import glob

from base import Cell, NetworkItem
from info_str import NAS_CONFIG
from utils import NAS_LOG, Logger, EvaScheduleItem


#### config params for eva alone ####
DATA_PATH = "./data/*"
# if you do not want to use all the data to train model, set the num here and -1 represent all
DATA_SIZE_FOR_SEARCH = 50000
DATA_RATIO_FOR_EVAL = 0.1

MODEL_PATH = "./model"
BATCH_SIZE = 50
INITIAL_LEARNING_RATE = 0.001


class DataSet:

    def __init__(self):
        self.all_train_data, self.all_train_label, self.train_data, self.train_label,\
            self.valid_data, self.valid_label, self.test_data, self.test_label = self.inputs()

    def inputs(self):
        '''
        Method for load data
        Returns:
            train_data, train_label, valid_data, valid_label, test_data, test_label
        '''
        # TODO read your data here, you must give train_data, train_label, test_data, test_label here
        all_train_data = []
        all_train_label = []
        train_data = []
        train_label = []
        test_data = []
        test_label = []

        return all_train_data, all_train_label, train_data, train_label, valid_data, valid_label, test_data, test_label

    def get_train_data(self, data_size):
        return self.train_data[:data_size], self.train_label[:data_size]

    def process(self, x):
        # TODO before you feed the data into computing graph, define the preprocess of data here
        return x


class DataFlowGraph:
    def __init__(self, task_item):
        """
        DataFlowGraph object and task_item are one-to-one correspondent
        """
        self.input_shape = [None, None, None, 3]
        self.output_shape = [None, None, None, 3]

        self.task_item = task_item
        self.net_item = task_item.network_item
        self.pre_block = task_item.pre_block
        self.cur_block_id = len(self.pre_block)
        self.ft_sign = task_item.ft_sign
        self.is_bestNN = task_item.is_bestNN
        
        self.repeat_num = NAS_CONFIG['nas_main']['repeat_num']
        # we add a pooling layer for last repeat block if the following signal is set true
        self.use_pooling_blk_end = NAS_CONFIG['nas_main']['link_node']
        
        # need to feed sth. when training
        self.data_x, self.data_y = None, None
        self.train_flag = False
        self.run_ops = {}  # what sess.run
        self.saver = None

    def _find_ends(self):
        if self.cur_block_id > 0 and self.net_item:  # if there are some previous blocks and not in retrain mode
            self._load_pre_model()
            graph = tf.get_default_graph()
            data_x = graph.get_tensor_by_name("input:0")
            data_y = graph.get_tensor_by_name("label:0")
            train_flag = graph.get_tensor_by_name("train_flag:0")
            mid_plug = graph.get_tensor_by_name("block{}/last_layer:0".format(self.cur_block_id - 1))
            if not self.ft_sign:
                mid_plug = tf.stop_gradient(mid_plug, name="stop_gradient")
        else:  # if it's the first block or in retrain mode
            data_x = tf.placeholder(tf.float32, self.input_shape, name='input')
            data_y = tf.placeholder(tf.float32, self.output_shape, name="label")
            train_flag = tf.placeholder(tf.bool, name='train_flag')
            mid_plug = tf.identity(data_x)
        return data_x, data_y, train_flag, mid_plug

    def _construct_graph(self):
        tf.reset_default_graph()

        self.data_x, self.data_y, self.train_flag, mid_plug = self._find_ends()
        
        blks = [[self.net_item.graph, self.net_item.cell_list]]
        mid_plug = self._construct_nblks(mid_plug, blks, self.cur_block_id)
        
        logits = tf.nn.dropout(mid_plug, keep_prob=1.0)
        model_output = tf.layers.conv2d(logits, 3, 3, padding='same', name="model_output", use_bias=False)
        
        #### here is output of model ####
        pred_img = tf.identity(model_output, name="pred_img")
        
        global_step = tf.Variable(0, trainable=False, name='global_step' + str(self.cur_block_id))
        accuracy = self._cal_accuracy(pred_img, self.data_y)
        loss = self._loss(pred_img, self.data_y)
        train_op = self._train_op(global_step, loss)

    def _cal_accuracy(self, logits, labels):
        # TODO change here for the way of calculating acc
        self.run_ops['acc'] = accuracy
        return accuracy

    def _loss(self, logits, labels):
        #  TODO change here for the way of calculating loss
        self.run_ops['loss'] = loss
        return loss

    def _train_op(self, global_step, loss):
        #  TODO you can change here to use another optimizer
        opt = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE, name='Momentum' + str(self.cur_block_id))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss, name="train_op")
        self.run_ops['train_op'] = train_op
        return train_op

    def _cal_multi_target(self, precision):
        # TODO change here for target calculating
        target = precision
        return target

    @staticmethod
    def _pad(inputs1, inputs2):
        # padding
        a = tf.shape(inputs2)[1]
        b = tf.shape(inputs1)[1]
        pad = tf.abs(tf.subtract(a, b))
        output = tf.where(tf.greater(a, b), 
                          tf.concat([tf.pad(inputs1, [[0, 0], [0, pad], [0, pad], [0, 0]]), inputs2], 3),
                          tf.concat([inputs1, tf.pad(inputs2, [[0, 0], [0, pad], [0, pad], [0, 0]])], 3))
        return output

    @staticmethod
    def _recode_repeat_blk(graph_full, cell_list, repeat_num):
        new_graph = [] + graph_full
        new_cell_list = [] + cell_list
        add = 0
        for i in range(repeat_num - 1):
            new_cell_list += cell_list
            add += len(graph_full)
            for sub_list in graph_full:
                new_graph.append([x + add for x in sub_list])
        return new_graph, new_cell_list

    def _construct_blk(self, blk_input, graph_full, cell_list, train_flag, blk_id):
        topo_order = self._toposort(graph_full)
        nodelen = len(graph_full)

        # input list for every cell in network
        inputs = [blk_input for _ in range(nodelen)]
        # bool list for whether this cell has already got input or not
        getinput = [False for _ in range(nodelen)]
        
        getinput[0] = True
        with tf.variable_scope('block' + str(blk_id)) as scope:
            for node in topo_order:
                layer = self._make_layer(inputs[node], cell_list[node], node, train_flag)
                # update inputs information of the cells below this cell
                for j in graph_full[node]:
                    # if this cell already got inputs from other cells precedes it
                    if getinput[j]:
                        inputs[j] = self._pad(inputs[j], layer)
                    else:
                        inputs[j] = layer
                        getinput[j] = True
            # give last layer a name
            last_layer = tf.identity(layer, name="last_layer")
        return last_layer

    def _construct_nblks(self, mid_plug, blks, first_blk_id):
        blk_id = first_blk_id
        for blk in blks:
            graph_full, cell_list = blk
            graph_full, cell_list = self._recode_repeat_blk(graph_full, cell_list, self.repeat_num)
            
            # add the last node
            graph_full = graph_full + [[]]
            if self.use_pooling_blk_end:
                cell_list = cell_list + [Cell('pooling', 'max', 2)]
            else:
                cell_list = cell_list + [Cell('id', 'max', 1)]
            
            mid_plug = self._construct_blk(mid_plug, graph_full, cell_list, self.train_flag, blk_id)

            blk_id += 1
        return mid_plug

    def _toposort(self, graph):
        node_len = len(graph)
        in_degrees = dict((u, 0) for u in range(node_len))
        for u in range(node_len):
            for v in graph[u]:
                in_degrees[v] += 1
        queue = [u for u in range(node_len) if in_degrees[u] == 0]
        result = []
        while queue:
            u = queue.pop()
            result.append(u)
            for v in graph[u]:
                in_degrees[v] -= 1
                if in_degrees[v] == 0:
                    queue.append(v)
        return result

    def _make_layer(self, inputs, cell, node, train_flag):
        if cell.type == 'conv':
            layer = self._makeconv(inputs, cell, node, train_flag)
        elif cell.type == 'pooling':
            layer = self._makepool(inputs, cell)
        elif cell.type == 'id':
            layer = tf.identity(inputs)
        elif cell.type == 'sep_conv':
            layer = self._makesep_conv(inputs, cell, node, train_flag)
        # TODO add any other new operations here
        #  use the form as shown above
        #  '''elif cell.type == 'operation_name':
        #         layer = self._name_your_function_here(inputs, cell, node)'''
        #  The "_name_your_function_here" is a function take (inputs, cell, node) or any other needed parameter as
        #  input, and output the corresponding tensor calculated use tensorflow, see self._makeconv as an example.
        #  The "inputs" is the input tensor, and "cell" is the hyper parameters for building this layer, given by
        #  class Cell(). The "node" is the index of this layer, mainly for the nomination of the output tensor.
        else:
            assert False, "Wrong cell type!"
        return layer

    def _name_your_function_here(self, inputs, cell, node):
        """
        the operation defined by user,
        Args:
            inputs: the input tensor of this operation
            cell: Class Cell(), hyper parameters for building this layer
            node: int, the index of this operation
        Returns:
            layer: the output tensor
        """
        # TODO add your function here if any new operation was added, see _makeconv as an example
        return

    def _makeconv(self, x, hplist, node, train_flag):
        with tf.variable_scope('conv' + str(node)) as scope:
            inputdim = x.shape[3]
            kernel = self._get_variable('weights',
                                        shape=[hplist.kernel_size, hplist.kernel_size, inputdim, hplist.filter_size])
            x = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._get_variable('biases', hplist.filter_size)
            x = self._batch_norm(tf.nn.bias_add(x, biases), train_flag)
            x = self._activation_layer(hplist.activation, x, scope)
        return x

    def _makesep_conv(self, inputs, hplist, node, train_flag):
        with tf.variable_scope('conv' + str(node)) as scope:
            inputdim = inputs.shape[3]
            dfilter = self._get_variable('weights', shape=[hplist.kernel_size, hplist.kernel_size, inputdim, 1])
            pfilter = self._get_variable('pointwise_filter', [1, 1, inputdim, hplist.filter_size])
            conv = tf.nn.separable_conv2d(inputs, dfilter, pfilter, strides=[1, 1, 1, 1], padding='SAME')
            biases = self._get_variable('biases', hplist.filter_size)
            bn = self._batch_norm(tf.nn.bias_add(conv, biases), train_flag)
            conv_layer = self._activation_layer(hplist.activation, bn, scope)
        return conv_layer

    def _batch_norm(self, input, train_flag):
        return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                            updates_collections=None, is_training=train_flag)

    def _get_variable(self, name, shape):
        if name == "weights":
            ini = tf.contrib.keras.initializers.he_normal()
        else:
            ini = tf.constant_initializer(0.0)
        return tf.get_variable(name, shape, initializer=ini)

    def _activation_layer(self, type, inputs, scope):
        if type == 'relu':
            layer = tf.nn.relu(inputs, name=scope.name)
        elif type == 'relu6':
            layer = tf.nn.relu6(inputs, name=scope.name)
        elif type == 'tanh':
            layer = tf.tanh(inputs, name=scope.name)
        elif type == 'sigmoid':
            layer = tf.sigmoid(inputs, name=scope.name)
        elif type == 'leakyrelu':
            layer = tf.nn.leaky_relu(inputs, name=scope.name)
        else:
            layer = tf.identity(inputs, name=scope.name)
        return layer

    def _makepool(self, inputs, hplist):
        if hplist.pooling_type == 'avg':
            return tf.nn.avg_pool(inputs, ksize=[1, hplist.kernel_size, hplist.kernel_size, 1],
                                  strides=[1, hplist.kernel_size, hplist.kernel_size, 1], padding='SAME')
        elif hplist.pooling_type == 'max':
            return tf.nn.max_pool(inputs, ksize=[1, hplist.kernel_size, hplist.kernel_size, 1],
                                  strides=[1, hplist.kernel_size, hplist.kernel_size, 1], padding='SAME')
        elif hplist.pooling_type == 'global':
            return tf.reduce_mean(inputs, [1, 2], keep_dims=True)

    def _makedense(self, inputs, hplist, with_bias=True):
        inputs = tf.reshape(inputs, [BATCH_SIZE, -1])
        for i, neural_num in enumerate(hplist[1]):
            with tf.variable_scope('block' + str(self.cur_block_id) + 'dense' + str(i)) as scope:
                weights = self._get_variable('weights', shape=[inputs.shape[-1], neural_num])
                mul = tf.matmul(inputs, weights)
                if with_bias:
                    biases = self._get_variable('biases', [neural_num])
                    mul += biases
                if neural_num == self.output_shape[-1]:
                    local3 = self._activation_layer('', mul, scope)
                else:
                    local3 = self._activation_layer(hplist[2], mul, scope)
            inputs = local3
        return inputs

    def _load_pre_model(self):  # for evaluate 
        front_blk_model_path = os.path.join(MODEL_PATH, 'model' + str(self.cur_block_id-1))
        assert os.path.exists(front_blk_model_path+".meta")
        self.saver = tf.train.import_meta_graph(front_blk_model_path+".meta")


class Evaluator:
    def __init__(self):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        self.log = ''
        self.data_set = DataSet()
        self.epoch = 0
        self.data_size = 0

    def _set_epoch(self, e):
        self.epoch = e
        return self.epoch

    def _set_data_size(self, num):
        self.data_size = num
        return self.data_size

    def evaluate(self, task_item):
        self._log_item_info(task_item)

        computing_graph = DataFlowGraph(task_item)
        computing_graph._construct_graph()

        score = self._train(computing_graph)

        score = self._test(computing_graph)
        
        NAS_LOG = Logger()
        NAS_LOG << ('eva_eva', self.log)
        return score

    def _train(self, compute_graph):
        epoch_log = ""
        valid_acc = 0
        for ep in range(self.epoch):
            # get the data

            # trian steps

            # valid steps

            self.log += epoch_log

        return valid_acc

    def _test(self, compute_graph):
        acc = 0

        return acc

    def _log_item_info(self, task_item):
        """
        record the info of network we will train
        we write the eva info in self.log and write it into the eva file once at the end of subprocess
        Args:
            task_item: one instance that represent one task and it store all itself info
        """
        self.log = ''  # reset the self.log
        if task_item.network_item:  # not in retrain mode
            self.log += "-"*20+"blk_id:"+str(len(task_item.pre_block))+" nn_id:"+str(task_item.nn_id)\
                        +" item_id:"+str(task_item.network_item.id)+"-"*20+'\n'
            for block in task_item.pre_block:
                self.log += str(block.graph) + str(block.cell_list) + '\n'
            self.log += str(task_item.network_item.graph) +\
                        str(task_item.network_item.cell_list) + '\n'
        else:  # in retrain mode
            self.log += "-"*20+"retrain"+"-"*20+'\n'
            for block in task_item.pre_block:
                self.log += str(block.graph) + str(block.cell_list) + '\n'


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    eval = Evaluator()
    cur_data_size = eval._set_data_size(100)
    cur_epoch = eval._set_epoch(1)

    # simulate the search process
    graph_full = [[1]]
    cell_list = [Cell('conv', 64, 3, 'leakyrelu')]
    network1 = NetworkItem(0, graph_full, cell_list, "")
    graph_full = [[1]]
    cell_list = [Cell('conv', 128, 3, 'relu6')]
    network2 = NetworkItem(1, graph_full, cell_list, "")

    task_item = EvaScheduleItem(nn_id=0, alig_id=0, graph_template=[], item=network1,\
         pre_blk=[], ft_sign=True, bestNN=True, rd=0, nn_left=0, spl_batch_num=6, epoch=cur_epoch, data_size=cur_data_size)
    e = eval.evaluate(task_item)

    task_item = EvaScheduleItem(nn_id=0, alig_id=0, graph_template=[], item=network2,\
         pre_blk=[network1], ft_sign=True, bestNN=True, rd=0, nn_left=0, spl_batch_num=6, epoch=cur_epoch, data_size=cur_data_size)
    e = eval.evaluate(task_item)

    task_item = EvaScheduleItem(nn_id=0, alig_id=0, graph_template=[], item=None,\
         pre_blk=[network1, network2], ft_sign=True, bestNN=True, rd=0, nn_left=0, spl_batch_num=6, epoch=cur_epoch, data_size=cur_data_size)
    e = eval.evaluate(task_item)

