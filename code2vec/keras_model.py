import tensorflow as tf
try:
    import tensorflow.python.keras as keras
    from tensorflow.python.keras import layers
    import tensorflow.python.keras.backend as K
except:
    import tensorflow.keras as keras
    from tensorflow.keras import layers
    import tensorflow.keras.backend as K
from typing import Optional
from code2vec.config import Config
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

from code2vec.keras_attention_layer import Attention_layer
# from collections import namedtuple
# from model_base import Code2VecModelBaseoptim
from code2vec.keras_topk_word_predictions_layer import TopKWordPredictionsLayer

class Code2VecModel():
    def __init__(self,config: Config):
        self.keras_train_model: Optional[keras.Model] = None
        self.keras_train_model_1: Optional[keras.Model] = None
        self.keras_eval_model: Optional[keras.Model] = None
#
        self._checkpoint: Optional[tf.train.Checkpoint] = None
        self._checkpoint_manager: Optional[tf.train.CheckpointManager] = None
        # super(Code2VecModel, self).__init__(config)
        self.config = config

    def _create_keras_model(self):
        # model_input layer
        # context_valid_mask: check whether the context is padding
        # (None, max_contents)
        path_source_token_input = layers.Input((self.config.MAX_CONTEXTS,), dtype=tf.int32)
        path_input = layers.Input((self.config.MAX_CONTEXTS,), dtype=tf.int32)
        path_target_token_input = layers.Input((self.config.MAX_CONTEXTS,), dtype=tf.int32)
        context_valid_mask = layers.Input((self.config.MAX_CONTEXTS,))

        # path embedding layer
        # (None, max_contents) -> (None,max_contents,path_embedding_size)
        paths_embedded = layers.Embedding(
            self.config.path_vocab_size, self.config.PATH_EMBEDDINGS_SIZE, name = 'path_embedding'
        )(path_input)

        # terminal embedding layer
        # (None, max_contents) -> (None,max_contents,token_embedding_size)
        token_embedding_shared_layer = layers.Embedding(
            self.config.token_vocab_size, self.config.TOKEN_EMBEDDINGS_SIZE, name = 'token_embedding'
        )

        path_source_token_embedded = token_embedding_shared_layer(path_source_token_input)
        path_target_token_embedded = token_embedding_shared_layer(path_target_token_input)

        # concatenate layer: paths -> [source, path, target]
        # [3 * (None,max_contents, token_embedding_size)] -> (None, max_contents,3*embedding_size)
        context_embedded = layers.Concatenate()([path_source_token_embedded, paths_embedded, path_target_token_embedded])
        context_embedded = layers.Dropout(1 - self.config.DROPOUT_KEEP_RATE)(context_embedded)

        # Dense layer: (None,max_contents,3*embedding_size) -> (None,max_contents, code_vector_size)
        context_after_dense = layers.TimeDistributed(
            layers.Dense(self.config.CODE_VECTOR_SIZE, use_bias=False, activation='tanh')
        )(context_embedded)

        # attention layer:  (None, max_contents,code_vector_size) -> (None,code_vector_size)
        code_vectors, attention_weights = Attention_layer(name='attention')(
            [context_after_dense, context_valid_mask]
        )

        # # problem encoding
        # problem_desc = layers.Input((self.config.CODE_VECTOR_SIZE,),dtype=tf.float32)
        #
        # #Wrong cases encoding
        # problem_cases = layers.Input((self.config.MAX_PROBLEM_CASES,),dtype=tf.float32)

        # all_encoding = layers.concatenate()([code_vectors,problem_desc,problem_cases])

        # Dense layer: softmax
        # (None,code_vector_size) -> (None,target_vocabs_size)
        target_index = layers.Dense(
            self.config.categories, use_bias = False, activation = 'sigmoid', name = 'target_index'
        )(code_vectors)

        # wrap into keras model
        inputs = [path_source_token_input, path_input, path_target_token_input, context_valid_mask]
        self.keras_train_model = keras.Model(inputs = inputs, outputs = target_index)

        problem_input = layers.Input((self.config.PROBLEM_INFO,),dtype=tf.int32)
        problem_embedding = layers.Embedding(self.config.problem_vocab_size,self.config.problem_embeddings_size,name='problem_embedding')
        problem_input_embedded = problem_embedding(problem_input)
        problem_embedded = layers.Flatten()(problem_input_embedded)
        problem_code_vectors = layers.Concatenate()([code_vectors,problem_embedded])
        problem_code_vectors = tf.nn.sigmoid(problem_code_vectors)
        target_index_1 = layers.Dense(
            self.config.categories, use_bias=False, activation='sigmoid', name='target_index1'
        )(problem_code_vectors)
        inputs_1 = [inputs,problem_input]
        self.keras_train_model_1 = keras.Model(inputs = inputs_1, outputs = target_index_1)

        submission_testCases_input = layers.Input((self.config.MAX_TESTCASES_LIMIT,), dtype=tf.float32)
        problem_code_testCase_vectors = layers.Concatenate()([problem_code_vectors,submission_testCases_input])

        target_index_2 = layers.Dense(
            self.config.categories, use_bias=False, activation='sigmoid', name='target_index2'
        )(problem_code_testCase_vectors)
        inputs_2 = [inputs_1,submission_testCases_input]
        self.keras_train_model_2 = keras.Model(inputs = inputs_2, outputs = target_index_2)

        """
            下面是用C2AE分类器进行分类的模型
        """
        Fx = layers.Dense(
            200, use_bias=True,activation=None,name='Fx'
        )(problem_code_testCase_vectors)

        Fx_relu = layers.LeakyReLU(alpha=0.1)(Fx)

        Fx_dropout = layers.Dropout(0.5)(Fx_relu)

        targets_input = layers.Input((self.config.categories,), dtype=tf.float32)
        targets_hidden = layers.Dense(
            200, use_bias=True,activation=None,name='targets_hidden'
        )(targets_input)
        targets_hidden_relu = layers.LeakyReLU(alpha=0.1)(targets_hidden)
        targets_hidden_dropout = layers.Dropout(0.5)(targets_hidden_relu)
        targets_output = layers.Dense(
            200, use_bias=True,activation=None,name='targets_embedding'
        )(targets_hidden_dropout)
        targets_output_relu = layers.LeakyReLU(alpha=0.1)(targets_output)
        targets_output_dropout = layers.Dropout(0.5)(targets_output_relu)

        targets_loss = layers.subtract([targets_output_dropout,Fx_dropout],name='targets_loss')

        Fd1 = layers.Dense(
            200, use_bias=True,activation=None,name='Fd1'
        )(Fx_dropout)

        Fd_relu1 = layers.LeakyReLU(alpha=0.1)(Fd1)
        Fd_dropout1 = layers.Dropout(0.5)(Fd_relu1)

        Fd = layers.Dense(
            200, use_bias=True, activation=None, name='Fd'
        )(Fd_dropout1)

        Fd_relu = layers.LeakyReLU(alpha=0.1)(Fd)
        Fd_dropout = layers.Dropout(0.5)(Fd_relu)

        target_index_3 = layers.Dense(
            self.config.categories, use_bias=True,activation='sigmoid',name='target_index_3'
        )(Fd_dropout)

        inputs_3 = [inputs_2, targets_input]
        self.keras_train_model_3 = keras.Model(inputs = inputs_3, outputs = [target_index_3,targets_loss])


        print("------------------create_keras_model Done.-------------------------")

    @classmethod
    def _create_optimizer(cls):
        return tf.keras.optimizers.Adam()

    def _comile_keras_model(self, optimizer=None):
        if optimizer is None:
            optimizer = self.keras_train_model.optimizer
        if optimizer is None:
            optimizer = self._create_optimizer()

        def getPrecision(y_true, y_pred):
            y_pred1 = y_pred
            row_dif = K.cast(K.sum(K.round(K.clip(y_true * (1 - y_pred1) + (1-y_true) * y_pred1,0,1)), axis=1) > K.epsilon(),'float32')
            dif = K.sum(K.round(row_dif))

            row_equ = K.cast(K.abs(K.sum(K.round(K.clip(y_true * y_pred1 + (1-y_true) * (1 - y_pred1),0,1)), axis=1) - self.config.categories) < K.epsilon(),'float32')
            equ = K.sum(K.round(row_equ))

            return equ / (equ + dif + K.epsilon())

        def micro_getPrecsion(y_true, y_pred):
            TP = tf.reduce_sum(y_true * tf.round(y_pred))
            TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
            FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
            FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
            precision = TP / (TP + FP + K.epsilon())
            # recall = TP / (TP + FN + K.epsilon())
            return precision

        def micro_getRecall(y_true, y_pred):
            TP = tf.reduce_sum(y_true * tf.round(y_pred))
            TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
            FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
            FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
            precision = TP / (TP + FP + K.epsilon())
            recall = TP / (TP + FN + K.epsilon())
            return recall

        # F1-score评价指标
        def micro_F1score(y_true, y_pred):
            TP = tf.reduce_sum(y_true * tf.round(y_pred))
            TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
            FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
            FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
            precision = TP / (TP + FP + K.epsilon())
            recall = TP / (TP + FN + K.epsilon())
            F1score = 2 * precision * recall / (precision + recall + K.epsilon())
            return F1score

        def macro_getPrecison(y_true, y_pred):
            col_TP = K.sum(y_true * K.round(y_pred), axis=0)
            col_TN = K.sum((1 - y_true) * (1 - K.round(y_pred)), axis=0)
            col_FP = K.sum((1 - y_true) * K.round(y_pred), axis=0)
            col_FN = K.sum(y_true * (1 - K.round(y_pred)), axis=0)
            precsion = K.mean(col_TP / (col_TP + col_FP + K.epsilon()))
            return precsion

        def macro_getRecall(y_true, y_pred):
            # print(y_true)
            row_TP = K.sum(y_true * K.round(y_pred), axis=0)
            row_TN = K.sum((1 - y_true) * (1 - K.round(y_pred)), axis=0)
            row_FP = K.sum((1 - y_true) * K.round(y_pred), axis=0)
            row_FN = K.sum(y_true * (1 - K.round(y_pred)), axis=0)
            recall = K.mean(row_TP / (row_TP + row_FN + K.epsilon()))
            return recall


        def macro_getF1score(y_true, y_pred):
            precision = macro_getPrecison(y_true, y_pred)
            recall = macro_getRecall(y_true, y_pred)
            F1score = 2 * precision * recall / (precision + recall + K.epsilon())
            return F1score

            # F1-score评价指标

        # def macro_F1score(y_true, y_pred):
        #     TP = tf.reduce_sum(y_true * tf.round(y_pred))
        #     TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
        #     FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
        #     FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
        #     precision = TP / (TP + FP)
        #     recall = TP / (TP + FN)
        #     F1score = 2 * precision * recall / (precision + recall)
        #     return F1score

        def cross_loss(y_true, y_pred):

            cross_loss = tf.add(tf.math.log(1e-10 + y_pred) * y_true, tf.math.log(1e-10 + (1 - y_pred)) * (1 - y_true))
            cross_entropy_label = -1 * tf.reduce_mean(tf.reduce_sum(cross_loss, 1))
            return cross_entropy_label

        self.keras_train_model.compile(
            # loss=tf.keras.losses.binary_crossentropy,
            loss=cross_loss,
            optimizer=optimizer,
            metrics=[getPrecision, micro_getPrecsion, micro_getRecall,micro_F1score,macro_getPrecison,macro_getRecall,macro_getF1score]
        )

        self.keras_train_model_1.compile(
            # loss=tf.keras.losses.binary_crossentropy,
            loss=cross_loss,
            optimizer=optimizer,
            metrics=[getPrecision, micro_getPrecsion, micro_getRecall,micro_F1score,macro_getPrecison,macro_getRecall,macro_getF1score]
        )

        self.keras_train_model_2.compile(
            # loss=tf.keras.losses.binary_crossentropy,
            loss=cross_loss,
            optimizer=optimizer,
            metrics=[getPrecision, micro_getPrecsion, micro_getRecall,micro_F1score,macro_getPrecison,macro_getRecall,macro_getF1score]
        )

        """
            C2AE损失函数：
                custom_loss: 
                    返回模型最后一层target_loss的平方和,这里y_true是随机设的
                binary_crossentropy:
                    返回输出的二分类交叉熵
        """
        def custom_loss(y_true,y_pred):
            return 10*tf.reduce_mean(tf.square(y_pred))

        # def custom_loss( Fx, Fe):
        #     Ix, Ie = tf.eye(tf.shape(Fx)[0]), tf.eye(tf.shape(Fe)[0])
        #     C1, C2, C3 = tf.abs(Fx - Fe), tf.matmul(Fx, tf.transpose(Fx)) - Ix, tf.matmul(Fe, tf.transpose(Fe)) - Ie
        #     return  tf.trace(tf.matmul(C1, tf.transpose(C1))) + self.config.lagrange_const * tf.trace(tf.matmul(C2, tf.transpose(C2))) + self.config.lagrange_const * tf.trace(tf.matmul(C3, tf.transpose(C3)))

        def output_loss( y_true,y_pred):
            Ei = 0.0
            i, cond = 0, 1
            while cond == 1:
                cond = tf.cond(i >= tf.shape(y_true)[0] - 1, lambda: 0, lambda: 1)
                # prediction_, Y_ = tf.slice(predictions, [i, 0], [1, self.config.labels_dim]), tf.slice(labels, [i, 0],[1,self.config.labels_dim])
                prediction_, Y_ = tf.slice(y_pred, [i, 0], [1, 11]), tf.slice(y_true, [i, 0],[1,11])
                zero, one = tf.constant(0, dtype=tf.float32), tf.constant(1, dtype=tf.float32)
                ones, zeros = tf.gather_nd(prediction_, tf.where(tf.equal(Y_, one))), tf.gather_nd(prediction_,tf.where(tf.equal(Y_,zero)))
                y1 = tf.reduce_sum(Y_)
                y0 = Y_.get_shape().as_list()[1] - y1
                temp = (1 / y1 * y0) * tf.reduce_sum(tf.exp(-(
                            tf.reduce_sum(ones) / tf.cast(tf.shape(ones)[0], tf.float32) - tf.reduce_sum(
                        zeros) / tf.cast(tf.shape(zeros)[0], tf.float32))))
                Ei += tf.cond(tf.logical_or(tf.math.is_inf(temp), tf.math.is_nan(temp)), lambda: tf.constant(0.0), lambda: temp)
                i += 1
            return 0.01*Ei

        def cross_loss(y_true, y_pred):
            cross_loss = tf.add(tf.math.log(1e-10 + y_pred) * y_true, tf.math.log(1e-10 + (1 - y_pred)) * (1 - y_true))
            cross_entropy_label = -1 * tf.reduce_mean(tf.reduce_sum(cross_loss, 1))
            return cross_entropy_label


        self.keras_train_model_3.compile(
            loss = {'target_index_3':cross_loss,'targets_loss':custom_loss},
            optimizer=optimizer,
            metrics={'target_index_3':[getPrecision, micro_getPrecsion, micro_getRecall,micro_F1score,macro_getPrecison,macro_getRecall,macro_getF1score]}
        )

        # self.keras_eval_model.compile(
        #     loss={'target_index':tf.nn.sigmoid_cross_entropy_with_logits(),'target_string':zero_loss},
        #     optimizer=optimizer
        # )

if __name__ == "__main__":
    config = Config(set_defaults=True, load_from_args=True, verify=False)
    model = Code2VecModel(config)
    model._create_keras_model()

    # visualize the model
    keras.utils.plot_model(model.keras_train_model, './code2VecTrainModel.png', show_shapes=True)
    # keras.utils.plot_model(model.keras_eval_model, './code2VecEvalModel.png', show_shapes=True)
