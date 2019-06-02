# -*- coding:utf-8 -*-
# import inspect
import tensorflow as tf
import numpy as np


def get_weight_variables(name, shape, weight_decay):
    weights = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    tf.add_to_collection('losses', regularizer(weights))
    return weights

def get_bias_variables(name, sz, weight_decay):
    bias = tf.get_variable(name, [sz, 1], initializer=tf.constant_initializer(np.zeros([sz])))
    # regularizer = tf.contrib.layers.l2_regularizer(0.000001)
    # tf.add_to_collection('losses', regularizer(bias))
    return bias


class Model(object):
    def __init__(self, is_training=True):
        self.keep_prob = 0.8 if is_training else 1.0
        self.lr = 0.001
        self.beta1=0.9
        self.user_seq_length = 15
        self.user_item_term_length = 10 
        self.user_query_term_length = 10 
        self.query_length = 10
        self.query_topcate_length = 3 
        self.query_leafcate_length = 3 
        self.embed_size_word = 64 
        self.weight_decay = 0.00001
        self.vocab_size = 280000
        self.is_training = is_training

    def get_batch_sample(self, placeholders):
        
        self.wide_feat_list = placeholders['wide_feat_list']
            
        self.user_item_seq_feat = placeholders['user_item_seq_feat']
        self.user_query_seq_feat = placeholders['user_query_seq_feat']
        self.user_query_item_feat = placeholders['user_query_item_feat']
        self.user_item_query_feat = placeholders['user_item_query_feat']
            
        self.query_item_query_feat = placeholders['query_item_query_feat']
        self.query_user_item_feat = placeholders['query_user_item_feat']
        self.query_feat = placeholders['query_feat']
            
        self.label_list = placeholders['label_list']
        self.config_keep_prob = prob = tf.placeholder_with_default(0.8, shape=(), name='keep_prob')

    def create_model_graph(self):
        ## 原先版本user termid和query termid的embedding参数不共享，当前版本改成共享的
        self.word_embed = word_embed  = tf.get_variable('word_embedding',[self.vocab_size, self.embed_size_word],trainable=True)
        
        
              
        with tf.variable_scope("query_embedding"):
            query_terms, query_topcate, query_leafcate  = tf.split(self.query_feat, [self.query_length, self.query_topcate_length, self.query_leafcate_length], 0)

            
            print ("query_terms:", query_terms.get_shape())
            
            # query term embedding =>  embedding_size_word x batch_size
            inputs_query_raw = tf.nn.embedding_lookup( self.word_embed, query_terms )
            print ("inputs_query_raw:", inputs_query_raw.get_shape())
            
            sign_query_feat = tf.sign(query_terms)
            
            input_num = tf.maximum( tf.reduce_sum( tf.sign(query_terms) ), 1)

            self.query_w2v_sum = query_w2v_sum = tf.transpose( tf.reduce_mean( inputs_query_raw[:input_num],axis=0 )  )
            
            print ( "query_w2v_sum:", query_w2v_sum.get_shape() )  #64
    


        def _myRNN( _X_S, hidden_size, name):
            with tf.variable_scope(name):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
                _LSTM_O, _LSTM_S = tf.contrib.rnn.static_rnn(lstm_cell, _X_S, dtype=tf.float32)
                print("_LSTM_O shape", _LSTM_O[-1].get_shape()) # batch_size * hidden_size 
                w_ip1 = get_weight_variables("rnn_inner_product_1_w", [64, hidden_size], self.weight_decay)
                b_ip1 = get_bias_variables("rnn_inner_product_1_b", 64, self.weight_decay)
                hidden_layer1 = tf.nn.tanh(tf.matmul(w_ip1, tf.transpose(_LSTM_O[-1])) + b_ip1)
                print("hidden_layer1 shape", hidden_layer1.get_shape())  #64 * batch_size

                _O = hidden_layer1
                return _O
            
            
            
        with tf.variable_scope("user_word_embedding"):

            user_item_seq_term_embed_list = []

            raw_step_termids_list = tf.split(self.user_item_seq_feat[-13 * 5:],  [13] * 5, 0)  
            for i in range( len( raw_step_termids_list ) ):
                item_terms, item_topcate, item_leafcate, time_delta = tf.split( raw_step_termids_list[i], [self.user_item_term_length, 1, 1, 1], 0)

                step_embedding = tf.nn.embedding_lookup( self.word_embed, item_terms )
                
                input_item_num = tf.maximum( tf.reduce_sum( tf.sign( item_terms ) ), 1)
                
                step_avg_embedding = tf.reduce_mean( step_embedding[ :input_item_num ],axis=0 )
                  
                user_item_seq_term_embed_list.append( step_avg_embedding  )
            
            print( "user_item_seq_term_embed_list", len(user_item_seq_term_embed_list) )
#             self.user_item_term_lstm_output = tf.transpose( tf.reduce_mean( user_item_seq_term_embed_list, 0 ) ) #取均值
            
            
            self.user_item_term_lstm_output = _myRNN(user_item_seq_term_embed_list, 64, "item_lstm_term")
            
            print("lstm self.user_item_term_lstm_output shape", self.user_item_term_lstm_output.get_shape())  #64 * batch_size

 


        with tf.variable_scope("user_item_query_embedding"):

            user_item_seq_term_embed_list = []

            raw_step_termids_list = tf.split( self.user_item_query_feat[:10 * 5],  [10] * 5, 0 )  
            for i in range( len( raw_step_termids_list ) ):
                query_terms = raw_step_termids_list[ i ][:5]

                step_embedding = tf.nn.embedding_lookup( self.word_embed, query_terms )
                
                input_query_num = tf.maximum( tf.reduce_sum( tf.sign( query_terms ) ), 1)
                
                step_avg_embedding = tf.reduce_mean( step_embedding[ :input_query_num ],axis=0 )
                  
                user_item_seq_term_embed_list.append( step_avg_embedding  )
            
            print( "user_item_query_term_embed_list", len(user_item_seq_term_embed_list) )
#             self.user_item_query_term_lstm_output = tf.transpose( tf.reduce_mean( user_item_seq_term_embed_list, 0 ) ) #取均值
            
            
            self.user_item_query_term_lstm_output = _myRNN(user_item_seq_term_embed_list, 64, "item_lstm_term")
            
            print("lstm self.user_item_query_term_lstm_output shape", self.user_item_query_term_lstm_output.get_shape())  #64 * batch_size

 

        with tf.variable_scope("user_query_item_embedding"):

            user_query_seq_term_embed_list = []

            raw_step_termids_list = tf.split( self.user_query_item_feat[-10 * 5:],  [10] * 5, 0 )  
            for i in range( len( raw_step_termids_list ) ):
                item_terms = raw_step_termids_list[ i ][:5]

                step_embedding = tf.nn.embedding_lookup( self.word_embed, item_terms )
                
                input_item_num = tf.maximum( tf.reduce_sum( tf.sign( item_terms ) ), 1)
                
                step_avg_embedding = tf.reduce_mean( step_embedding[ :input_item_num ],axis=0 )
                  
                user_query_seq_term_embed_list.append( step_avg_embedding  )
            
            print( "user_query_item_term_embed_list", len(user_query_seq_term_embed_list) )
#             self.user_query_item_term_lstm_output = tf.transpose( tf.reduce_mean( user_item_seq_term_embed_list, 0 ) ) #取均值
            
            
            self.user_query_item_term_lstm_output = _myRNN(user_query_seq_term_embed_list, 64, "item_lstm_term")
            
            print("lstm self.user_query_item_term_lstm_output shape", self.user_query_item_term_lstm_output.get_shape())  #64 * batch_size
        
        

    
        with tf.variable_scope("query_item_query_embedding"):

                item_query_terms = self.query_item_query_feat

                item_query_embedding = tf.nn.embedding_lookup( self.word_embed, item_query_terms )
                
                self.query_embedding = item_query_embedding
                
                input_item_num = tf.maximum( tf.reduce_sum( tf.sign( item_query_terms ) ), 1)
                
                avg_embedding = tf.reduce_mean( item_query_embedding[ :input_item_num ],axis=0 )
                  

                self.query_item_query_avg_output = tf.transpose( avg_embedding ) #取均值
            
            
                print("lstm self.query_item_query_avg_output shape", self.query_item_query_avg_output.get_shape())  #64 * batch_size
                
                
                
        with tf.variable_scope("query_user_item_embedding"):

                item_query_terms = self.query_user_item_feat

                item_query_embedding = tf.nn.embedding_lookup( self.word_embed, item_query_terms )
                
                input_item_num = tf.maximum( tf.reduce_sum( tf.sign( item_query_terms ) ), 1)
                
                avg_embedding = tf.reduce_mean( item_query_embedding[ :input_item_num ],axis=0 )
                  

                self.query_user_item_avg_output = tf.transpose( avg_embedding ) #取均值
            
            
                print("lstm self.query_user_item_avg_output shape", self.query_user_item_avg_output.get_shape())  #64 * batch_size


          

        with tf.variable_scope("user_query_seq_embedding"):
            user_query_seq_term_embed_list = []
            raw_step_query_termids_list = tf.split(self.user_query_seq_feat[-17 * 5:], [ 17 ] * 5, 0) #5个长度  
                
            for i in range( len(raw_step_query_termids_list) ):
                query_terms, query_topcate, query_leafcate, time_delta = tf.split( raw_step_query_termids_list[-i], [self.user_query_term_length, 3, 3, 1], 0)
                
                step_query_terms_embedding = tf.nn.embedding_lookup( self.word_embed, query_terms ) 
                input_item_num = tf.maximum( tf.reduce_sum( tf.sign( query_terms ) ), 1)
                step_query_terms_avg_embedding = tf.reduce_mean( step_query_terms_embedding[ :input_item_num ],axis=0 )
                user_query_seq_term_embed_list.append( step_query_terms_avg_embedding  )
                
            print( "user_query_seq_term_embed_list", len(user_query_seq_term_embed_list) )
#             self.user_query_term_lstm_output = tf.transpose( tf.reduce_mean( user_query_seq_term_embed_list, 0 ) ) #取均值
            
            
            self.user_query_term_lstm_output = _myRNN(user_query_seq_term_embed_list, 64, "query_lstm_term")
            
            print("lstm self.user_query_term_lstm_output shape", self.user_query_term_lstm_output.get_shape())  #64 * batch_size

            
        with tf.variable_scope("wide_full_connect"):
            w_wide_1 = get_weight_variables("wide_feat_1_w", [64, 81], self.weight_decay)
            b_wide_1 = get_bias_variables("wide_feat_1_b", 64, self.weight_decay)
            self.wide_hidden_layer1 = tf.nn.tanh(tf.nn.dropout(tf.matmul(w_wide_1, self.wide_feat_list) + b_wide_1, self.config_keep_prob))
            print ("wide_hidden_layer1:", self.wide_hidden_layer1.get_shape()) # 64 * batch_size
 


        with tf.variable_scope("concat_query_user"):
            with tf.variable_scope("concat_qu_term"):
                 
                    
                    
                qu_term_concat = tf.nn.dropout(tf.concat([
                    self.user_item_term_lstm_output,  \
                    self.user_query_term_lstm_output, \
                    self.query_w2v_sum, \
                    self.user_item_query_term_lstm_output, \
                    self.user_query_item_term_lstm_output,  \
                    self.query_item_query_avg_output,  \
                    self.query_user_item_avg_output\
                    ], 0), self.config_keep_prob)# U_qi_iq Q_ui_iq
                print("qu_term_concat shape ",qu_term_concat.get_shape())          
                w_qu_term_1 = get_weight_variables("w_qu_term_1", [64, 64*7], self.weight_decay)
      
    
    

                b_qu_term_1 = get_bias_variables("b_qu_term_1", 64, self.weight_decay)
                qu_term_hidden_layer1 = tf.nn.tanh(tf.nn.dropout(tf.matmul(w_qu_term_1, qu_term_concat) + b_qu_term_1, self.config_keep_prob))
                print("qu_term_hidden_layer1 shape ",qu_term_hidden_layer1.get_shape())   # 64 * batch_size

                
                
        with tf.variable_scope("concat_query_user_wide_all"): 
            deep_wide_concat = tf.concat([qu_term_hidden_layer1, self.wide_hidden_layer1], 0)
#             deep_wide_concat = qu_term_hidden_layer1
#             deep_wide_concat = self.wide_hidden_layer1
            w_dw_1 = get_weight_variables("deep_wide_feat_1_w", [64, 128], self.weight_decay)
            print("deep_wide_concat shape ",deep_wide_concat.get_shape())
    
    
            b_dw_1 = get_bias_variables("deep_wide_feat_1_b", 64, self.weight_decay)
            dw_hidden_layer1 = tf.nn.tanh(tf.nn.dropout(tf.matmul(w_dw_1, deep_wide_concat) + b_dw_1, self.config_keep_prob))

            w_dw_3 = get_weight_variables("deep_wide_feat_3_w", [1, 64], self.weight_decay)
            b_dw_3 = get_bias_variables("deep_wide_feat_3_b", 1, self.weight_decay)
            dw_hidden_layer3 = tf.matmul(w_dw_3, dw_hidden_layer1) + b_dw_3
            print("dw_hidden_layer3 shape ",dw_hidden_layer3.get_shape())   # 1 * batch_size

            self.rst_score = dw_hidden_layer3 
            print("self.rst_score shape ",self.rst_score.get_shape())   # 1 * batch_size

            self.global_res = global_res = tf.squeeze(dw_hidden_layer3)
            self.label  = tf.squeeze(self.label_list)

        print('net forward over!')

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.label, logits=self.global_res)) + tf.add_n(tf.get_collection('losses'))


        with tf.name_scope('AUC'):
            _, self.AUC_real = tf.contrib.metrics.streaming_auc(
                tf.nn.sigmoid(self.global_res), self.label)
        with tf.name_scope('summary'):
            self.summary = [
                tf.summary.histogram('global_res', self.global_res),
                tf.summary.scalar('AUC_real', self.AUC_real),
                tf.summary.scalar('loss', self.loss)
            ]

    def signature_def(self):
        tensor_info_x = tf.saved_model.utils.build_tensor_info(self.org_feature)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(self.rst_score)
        tensor_info_d = tf.saved_model.utils.build_tensor_info(self.config_keep_prob)
        
        tensor_info_embedding_for_exp = tf.saved_model.utils.build_tensor_info( self.query_embedding )

        return tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'inputs': tensor_info_x,'keep_prob': tensor_info_d},
                outputs={'score': tensor_info_y, "embedding_vector": tensor_info_embedding_for_exp},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    def build(self, placeholders):
        self.get_batch_sample(placeholders)
        self.create_model_graph()
    def assign_new_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
