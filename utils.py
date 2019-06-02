import numpy as np
import tensorflow as tf
def get_placeholder():
    
    placeholders = {
        'wide_feat_list': tf.placeholder(tf.float32, shape = [81, None]),
        'user_item_seq_feat': tf.placeholder(tf.int32, shape = [195, None]),
        'query_feat': tf.placeholder(tf.int32, shape=[16,None]),
        'user_query_seq_feat': tf.placeholder(tf.int32, shape=[170, None]),
        'query_item_query_feat': tf.placeholder(tf.int32, shape=[100, None]),
        'user_query_item_feat': tf.placeholder(tf.int32, shape=[100, None]),
        'user_item_query_feat': tf.placeholder(tf.int32, shape=[150, None]), 
        'query_user_item_feat': tf.placeholder(tf.int32, shape=[100, None]),
        'label_list': tf.placeholder(tf.float32)
    }

    return placeholders

def update_placeholder(placeholders, batch):
    feed_dict={}
    feed_dict.update({placeholders['wide_feat_list']:batch[0]})
    feed_dict.update({placeholders['user_item_seq_feat']:batch[1]})
    feed_dict.update({placeholders['query_feat']:batch[2]})
    feed_dict.update({placeholders['user_query_seq_feat']:batch[3]})
    feed_dict.update({placeholders['query_item_query_feat']:batch[4]})
    feed_dict.update({placeholders['user_query_item_feat']:batch[5]})
    feed_dict.update({placeholders['user_item_query_feat']:batch[6]})
    feed_dict.update({placeholders['query_user_item_feat']:batch[7]})
    feed_dict.update({placeholders['label_list']:batch[8]})
    return feed_dict

class data_precess(object):
    def __init__(self, path):
        self.features_list = []
        self.label_list = []
        with open(path) as fr:
            for line in fr:
                line = line.split()
                features, label = line[0], line[1]
                self.features_list.append(features)
                self.label_list.append(label)
        self.data_size = len(self.features_list)
  
#generate transNet training batches
    def batch_iter(self, batch_size):
        data_size = self.data_size
        shuffle_indices = np.arange(data_size)
        start_index = 0
        batch_id = 0
        end_index = min(start_index+batch_size, data_size)
        wide_feat_list = []
        user_item_seq_feat = []
        query_feat = []
        user_query_seq_feat = []
        query_item_query_feat = []
        user_query_item_feat = []
        user_item_query_feat = []
        query_user_item_feat = []
        label_list = []
        while start_index < data_size:
            for i in range(start_index, end_index):
                feas_split = self.features_list[shuffle_indices[i]].split(',')
                wide_feat_list.append([float(s) for s in feas_split[:81]])
                user_item_seq_feat.append([int(s) for s in feas_split[81:276]])
                query_feat.append([int(s) for s in feas_split[276:292]])
                user_query_seq_feat.append([int(s) for s in feas_split[292:462]])
                query_item_query_feat.append([int(s) for s in feas_split[462:562]])
                user_query_item_feat.append([int(s) for s in feas_split[562:662]])
                user_item_query_feat.append([int(s) for s in feas_split[662:812]])
                query_user_item_feat.append([int(s) for s in feas_split[812:]])
                label_list.append(float(self.label_list[shuffle_indices[i]]))
                
            batch_id += 1
            yield np.array(wide_feat_list).transpose(), np.array(user_item_seq_feat).transpose(), np.array(query_feat).transpose(),\
            np.array(user_query_seq_feat).transpose(), np.array(query_item_query_feat).transpose(), np.array(user_query_item_feat).transpose(), np.array(user_item_query_feat).transpose(), np.array(query_user_item_feat).transpose(), np.array(label_list).transpose()

            start_index = end_index
            end_index = min(start_index+batch_size, data_size)
