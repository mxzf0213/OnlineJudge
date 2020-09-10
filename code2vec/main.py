from code2vec.keras_model import Code2VecModel
from code2vec.config import Config
import pandas as pd
import os,sys
import re
import numpy as np
import tensorflow as tf
try:
    import tensorflow.python.keras as keras
    from tensorflow.python.keras.callbacks import TensorBoard
except:
    import tensorflow.keras as keras
    from tensorflow.keras.callbacks import TensorBoard
class Code2tags():
    def __init__(self, problem_info = True, submission_testCases = False, C2AE = False):
        self.config = Config(set_defaults=True, load_from_args=False, verify=False)
        self.model = Code2VecModel(self.config)
        self.base_path = 'code2vec/codeData2/'
        self.rawData_path = self.base_path + 'raw_data1'
        self.outData_path = self.base_path + 'outData'
        self.tempData_path = self.base_path + 'tempData'
        self.predict_code_path = self.base_path + 'predictCode'
        self.parsed_code_path = self.base_path + 'parsedCode'
        self.submission_details_path = self.base_path + 'submissions_details_data'
        self.modelSaved_path = 'code2vec/training'
        self.problem_info = problem_info
        self.cur_model = None
        self.submission_testCases = submission_testCases
        self.C2AE = C2AE
        self.node_types_dict = {}
        self.paths_dict = {}
        self.tokens_dict = {}
        self.already_get_hashed_table = False
        if self.C2AE:
            self.input_num = 8
        elif self.submission_testCases:
            self.input_num = 6
        elif self.problem_info:
            self.input_num = 5
        else:
            self.input_num = 4

    def preprocess_data(self):
        rawData_path = self.rawData_path
        outData_path = self.outData_path
        tempData_path = self.tempData_path
        if not os.path.exists(outData_path):
            os.mkdir(outData_path)
        if not os.path.exists(tempData_path):
            os.mkdir(tempData_path)

        pattern = r'[ ,.;]'
        map = {}
        global number
        number = 0
        def processLine(row):
            tag = row['校验']
            if pd.isnull(tag):
                return
            tag = str(tag)

            id = row['id']
            code = row['code']
            with open(os.path.join(tempData_path,id+'.c'),'w',encoding='utf-8') as f:
                f.write(code)

            tag1 = re.split(pattern,tag)
            result_tags = []
            for x in tag1:
                if x.isdigit() and int(x) != 0:
                    result_tags.append(int(x))
            map[id] = (result_tags,row['problem_id'])
            global number
            number = number+1

        for root, dirs, files in os.walk(rawData_path):
            for f in files:
                cur = pd.read_excel(os.path.join(rawData_path,f))
                cur.apply(processLine,axis=1)

        print("Extract Code and Tags({:d}) done!".format(number))

        ret = os.system(r'java -jar cli.jar pathContexts --lang c --project ' + self.tempData_path + ' --output ' + self.outData_path +
            r' --maxH 8 --maxW 2 --maxContexts ' +  str(self.config.MAX_CONTEXTS) + ' --maxTokens '+ str(self.config.MAX_TOKEN_VOCAB_SIZE) +
                        ' --maxPaths ' + str(self.config.MAX_PATH_VOCAB_SIZE))

        assert ret == 0
        print("Extract Code Paths done!")

        np.save(os.path.join(outData_path , 'dict.npy'),map)

        print("Saving Dict done!")

        if self.submission_testCases:
            self.preprocess_submissions()

    def preprocess_submissions(self):
        data_name = 'data.csv'
        curData_path = os.path.join(self.submission_details_path, data_name)
        cur_file = pd.read_csv(curData_path)
        map = {}
        def id2detailVector(row):
            cur_info = eval(row['info'])
            cur_id = row['id']
            cur_details = []
            if row['info'] == '{}':
                cur_details = [0 for i in range(self.config.MAX_TESTCASES_LIMIT)]
                map[cur_id] = cur_details
                return

            for _i,test_case in enumerate(cur_info['data']):
                if _i >= self.config.MAX_TESTCASES_LIMIT:
                    break
                cur_details.append(1 if int(test_case['result']) == 0 else -1)
            while len(cur_details) < 10:
                cur_details.append(0)
            map[cur_id] = cur_details
            return

        cur_file.apply(id2detailVector, axis=1)

        np.save(os.path.join(self.outData_path , 'testCases.npy'), map)
        print("Saving submissions id2testCases dict done")

    def check_data_distribution(self):
        submap = self.substractCategories()
        id2tags = np.load(os.path.join(self.outData_path, 'dict.npy'), allow_pickle=True).item()
        cnt = np.zeros((self.config.categories+1,))
        for key,value in id2tags.items():
            if len(value[0]) == 0:
                continue
            for x in value[0]:
                cnt[submap[x -1]] = cnt[submap[x -1]] + 1
            # cnt[list(map(lambda x: submap[x - 1] - 1, value[0]))] = cnt[submap[value[0]-1]] + 1
        print(cnt[1:])

    def substractCategories(self):
        """
        将稀疏的类别映射到稠密的类别
        当前<20的样本保持上一个样本的类别
        :return:
        """
        cur_map = [1,2,3,3,3,4,4,4,4,5,6,7,8,8,8,8,8,8,8,8,8,9,10,10,10,10,10,10,10,10,10,10,11]
        # test
        # test = [1,2,3,4]
        # print(list(map(lambda x: cur_map[x - 1] - 1, test)))
        return cur_map

    def load_data(self):
        substract_categories = self.config.substract_categories
        id2tags = np.load(os.path.join(self.outData_path,'dict.npy'),allow_pickle=True).item()
        testCases_details = np.load(os.path.join(self.outData_path,'testCases.npy'),allow_pickle=True).item()
        submap = self.substractCategories()

        Min = 999
        Max = 0
        Min1 = 999
        Max1 = 0
        source_tokens = []
        path_tokens = []
        target_tokens = []
        context_valid_masks = []
        problem_input = []
        targets = []
        submission_testCases = []
        unused_outputs = []
        for root, dirs, files in os.walk(self.outData_path):
            for f_name in files:
                if not f_name.startswith('path_contexts'):
                    continue
                with open(os.path.join(self.outData_path,f_name), 'r') as f:
                    p = f.readline()
                    while p:
                        line = p.split(' ')
                        Min = min(len(line) - 1, Min)
                        Max = max(len(line) - 1, Max)
                        id = line[0].split('\\')[-1].split('.')[0]
                        try:
                            if len(id2tags[id][0]) == 0:
                                p = f.readline()
                                continue
                        except:
                            p=f.readline()
                            continue
                        # cur_not = False
                        # for p in id2tags[id][0]:
                        #     if p == 23:
                        #         cur_not = True
                        # if cur_not == True:
                        #     p = f.readline()
                        #     continue
                        cur_targets = np.zeros((self.config.categories,))
                        if not substract_categories:
                            cur_targets[list(map(lambda x:x-1,id2tags[id][0]))] = 1
                        else:
                            cur_targets[list(map(lambda x: submap[x - 1] - 1, id2tags[id][0]))] = 1

                        targets.append(cur_targets)
                        unused_outputs.append([0 for i in range(400)])

                        submission_testCases.append(testCases_details[id])

                        cur_source_tokens = []
                        cur_path_tokens = []
                        cur_target_tokens = []
                        cur_context_masks = []
                        cur_problem_input = []
                        cur_problem_input.append(id2tags[id][1])
                        Min1 = min(id2tags[id][1], Min1)
                        Max1 = max(id2tags[id][1], Max1)

                        cnt = 0
                        for path in line[1:]:
                            source_token, path_token, target_token = map(int, path.split(','))
                            cur_source_tokens.append(source_token)
                            cur_path_tokens.append(path_token)
                            cur_target_tokens.append(target_token)
                            cur_context_masks.append(1)
                            cnt = cnt + 1
                        p = f.readline()
                        cur_source_tokens = np.pad(cur_source_tokens, (0, self.config.MAX_CONTEXTS - cnt), 'constant', constant_values=(0, 0))
                        cur_path_tokens = np.pad(cur_path_tokens, (0, self.config.MAX_CONTEXTS - cnt), 'constant', constant_values=(0, 0))
                        cur_target_tokens = np.pad(cur_target_tokens, (0, self.config.MAX_CONTEXTS - cnt), 'constant', constant_values=(0, 0))
                        cur_context_masks = np.pad(cur_context_masks, (0, self.config.MAX_CONTEXTS - cnt), 'constant', constant_values=(0, 0))
                        source_tokens.append(cur_source_tokens)
                        path_tokens.append(cur_path_tokens)
                        target_tokens.append(cur_target_tokens)
                        problem_input.append(cur_problem_input)
                        context_valid_masks.append(cur_context_masks)

        print("single path length range is {:d} to {:d}".format(Min, Max))
        print("problem id range is {:d} to {:d}".format(Min1, Max1))
        print("having {:d} path length".format(len(path_tokens)))

        if self.C2AE:
            return source_tokens, path_tokens, target_tokens, context_valid_masks, problem_input, submission_testCases,targets,targets,unused_outputs
        elif self.submission_testCases:
            return source_tokens, path_tokens, target_tokens, context_valid_masks, problem_input, submission_testCases, targets
        elif self.problem_info:
            return source_tokens, path_tokens, target_tokens, context_valid_masks, problem_input, targets
        else:
            return source_tokens, path_tokens, target_tokens, context_valid_masks, targets

    def load_data_2(self):
        numpy_path = os.path.join(self.base_path,'numpy_data')
        if self.C2AE:
            train_path = 'C2AE_train.npy'
            test_path = 'C2AE_test.npy'
        elif self.submission_testCases:
            train_path = 'problem_testCases_train.npy'
            test_path = 'problem_testCases_test.npy'
        elif self.problem_info:
            train_path = 'problem_info_train.npy'
            test_path = 'problem_info_test.npy'
        else:
            train_path = 'origin_train.npy'
            test_path = 'origin_test.npy'
        train_data = np.load(os.path.join(numpy_path,train_path), allow_pickle=True)
        test_data = np.load(os.path.join(numpy_path,test_path), allow_pickle=True)
        return train_data.tolist(), test_data.tolist()

    def train(self,inputs):
        self.model._create_keras_model()
        self.model._comile_keras_model()

        if self.C2AE:
            cur_model = self.model.keras_train_model_3
            modelSaved_path = os.path.join(self.modelSaved_path, 'C2AE_model')
        elif self.submission_testCases:
            cur_model = self.model.keras_train_model_2
            modelSaved_path = os.path.join(self.modelSaved_path, 'problem_testCases_model')
        elif self.problem_info:
            cur_model = self.model.keras_train_model_1
            modelSaved_path = os.path.join(self.modelSaved_path, 'problem_info_model')
        else:
            cur_model = self.model.keras_train_model
            modelSaved_path = os.path.join(self.modelSaved_path, 'origin_model')

        if not os.path.exists(modelSaved_path):
            os.mkdir(modelSaved_path)

        # keras.utils.plot_model(cur_model,os.path.join(modelSaved_path ,'model.png'),show_shapes=True)
        checkpoint_path = os.path.join(modelSaved_path,"cp-{epoch:04d}.ckpt")

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True,
                                                         period=self.config.SAVE_EVERY_EPOCHS)

        cur_model.save_weights(checkpoint_path.format(epoch=0))

        tensorboard = TensorBoard(log_dir=os.path.join(modelSaved_path,"logs"))

        # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        if not self.C2AE:
            history = cur_model.fit(inputs[:self.input_num], [inputs[self.input_num]],batch_size = self.config.TRAIN_BATCH_SIZE,
                                                   epochs = self.config.NUM_TRAIN_EPOCHS,callbacks=[cp_callback,tensorboard],
                                validation_split=1/3)
        else:
            history = cur_model.fit(inputs[:-2], inputs[-2:],
                                    batch_size=self.config.TRAIN_BATCH_SIZE,
                                    epochs=self.config.NUM_TRAIN_EPOCHS, callbacks=[cp_callback, tensorboard],
                                    validation_split=1 / 3)
        print("History:")
        print(history.history)

        self.cur_model = cur_model

    def load_savedModel(self,checkpoint_path):
        self.model._create_keras_model()
        self.model._comile_keras_model()

        if self.C2AE:
            cur_model = self.model.keras_train_model_3
        elif self.submission_testCases:
            cur_model = self.model.keras_train_model_2
        elif self.problem_info:
            cur_model = self.model.keras_train_model_1
        else:
            cur_model = self.model.keras_train_model
        cur_model.load_weights(checkpoint_path)
        self.cur_model = cur_model

    def predict(self,inputs):
        if self.cur_model == None:
            raise Exception("model not loaded")
        return self.cur_model.predict(inputs)

    def evaluate(self,inputs,targets):
        if self.cur_model == None:
            raise Exception("model not loaded")
        return self.cur_model.evaluate(inputs, targets, verbose=0)

    def get_hashed_table(self):
        node_types_dict = {}
        paths_dict = {}
        tokens_dict = {}
        node_types_path = os.path.join(self.outData_path,"node_types.csv")
        paths_path = os.path.join(self.outData_path,"paths.csv")
        tokens_path = os.path.join(self.outData_path,"tokens.csv")
        file1 = pd.read_csv(node_types_path)
        def node_types2dict(row):
            node_types_dict[row['node_type']] = row['id']
        file1.apply(node_types2dict,axis=1)
        file2 = pd.read_csv(paths_path)
        def paths2dict(row):
            paths_dict[row['path']] = row['id']
        file2.apply(paths2dict,axis=1)
        file3 = pd.read_csv(tokens_path)
        def tokens2dict(row):
            tokens_dict[row['token']] = row['id']
        file3.apply(tokens2dict,axis=1)

        self.node_types_dict = node_types_dict
        self.paths_dict = paths_dict
        self.tokens_dict = tokens_dict

    def predict_code(self, code = None, problem_id = None, submission_detail = None):
        if not self.already_get_hashed_table:
            self.get_hashed_table()
            self.already_get_hashed_table = True

        code_path = os.path.join(self.predict_code_path, 'main.c')

        if code == None:
            raise Exception("code should be str")

        with open(code_path,'w') as f:
            f.write(code)

        cur_order = r'java -jar code2vec/cli.jar pathContexts --lang c --project ' + self.predict_code_path + ' --output ' + self.parsed_code_path +\
        r' --maxH 8 --maxW 2 --maxContexts ' + str(self.config.MAX_CONTEXTS) + ' --maxTokens ' + str(self.config.MAX_TOKEN_VOCAB_SIZE) +\
            ' --maxPaths ' + str(self.config.MAX_PATH_VOCAB_SIZE)
        ret = os.system(cur_order)
        assert ret == 0

        node_types_path = os.path.join(self.parsed_code_path, "node_types.csv")
        paths_path = os.path.join(self.parsed_code_path, "paths.csv")
        tokens_path = os.path.join(self.parsed_code_path, "tokens.csv")
        code_path = os.path.join(self.parsed_code_path, "path_contexts_0.csv")

        file1 = pd.read_csv(node_types_path)
        file2 = pd.read_csv(paths_path)
        file3 = pd.read_csv(tokens_path)

        temp_node_types_dict = {}
        temp_paths_dict = {}
        temp_tokens_dict = {}
        def node_types2dict(row):
            temp_node_types_dict[row['id']] = row['node_type']
        file1.apply(node_types2dict,axis=1)
        def paths2dict(row):
            temp_paths_dict[row['id']] = row['path']
        file2.apply(paths2dict,axis=1)
        def tokens2dict(row):
            temp_tokens_dict[row['id']] = row['token']
        file3.apply(tokens2dict,axis=1)

        source_input = []
        path_input = []
        target_input = []
        context_valid_input = []

        file4 = open(code_path,"r")
        code = file4.readline().split(' ')[1:]
        cnt = 0

        for path in code:
            temp_source,temp_path,temp_target = map(int,path.split(','))
            try:
                real_source = self.tokens_dict[temp_tokens_dict[temp_source]]
                real_target = self.tokens_dict[temp_tokens_dict[temp_target]]
                real_path = self.paths_dict[' '.join(list(map(lambda x:str(self.node_types_dict[x]),
                                                        list(map(lambda x:temp_node_types_dict[int(x)],temp_paths_dict[temp_path].split(' '))))))]
                context_valid_input.append(1)
            except:
                # if the path cannot map into the vocabulary due to dataset reason or unknown cli.jar technical stargety
                # I currently choose to set value to zero to delete the path
                real_path = 0
                context_valid_input.append(0)
                real_source = 0
                real_target = 0

            source_input.append(real_source)
            path_input.append(real_path)
            target_input.append(real_target)
            # print(real_source,real_path,real_target)
            cnt = cnt + 1

        file4.close()

        source_input = [np.pad(source_input, (0, self.config.MAX_CONTEXTS - cnt), 'constant', constant_values=(0, 0))]
        path_input = [np.pad(path_input, (0, self.config.MAX_CONTEXTS - cnt), 'constant', constant_values=(0, 0))]
        target_input = [np.pad(target_input, (0, self.config.MAX_CONTEXTS - cnt), 'constant', constant_values=(0, 0))]
        context_valid_input = [np.pad(context_valid_input, (0, self.config.MAX_CONTEXTS - cnt), 'constant', constant_values=(0, 0))]

        if self.C2AE:
            ...
        elif self.submission_testCases:
            submission_detail_input = submission_detail
            problem_id_input = problem_id
            inputs = (source_input,path_input,target_input,context_valid_input,problem_id_input,submission_detail_input)
        elif self.problem_info:
            problem_id_input = problem_id
            inputs = (source_input, path_input, target_input, context_valid_input, problem_id_input)
        else:
            inputs = (source_input, path_input, target_input, context_valid_input)

        index2str = ['incorrect input strings','incorrect input variables','no output','incorrect output format','incorrect initialization',
                     'incorrect data types','incorrect data precision','incorrect loops','incorrect branches','incorrect logic','incorrect operators']

        results = self.predict(inputs).flatten().tolist()
        results_to_str = list(map(lambda x:(index2str[x[0]],x[1]),enumerate(results)))
        results_to_str.sort(key = lambda x:x[1], reverse=True)
        sorted_str = list(map(lambda x:x[0],results_to_str))
        sorted_prob = list(map(lambda x:x[1],results_to_str))
        return sorted_str,sorted_prob

    def batch_predict(self,code_csv_path,isReturnVector):
        if not self.already_get_hashed_table:
            self.get_hashed_table()
            self.already_get_hashed_table = True

        if isReturnVector:
            temp_model = keras.Model(inputs = self.cur_model.input, outputs = self.cur_model.get_layer('attention').output[0])
        else:
            temp_model = self.cur_model

        # cur_csv = pd.read_csv(code_csv_path,encoding='gbk')
        # data = cur_csv['code']
        # name = cur_csv['w']
        #
        # for i in range(len(data)):
        #     if(str(name[i]) == 'nan'):
        #         continue
        #     code_path = os.path.join(self.predict_code_path, str(name[i]))
        #     with open(code_path + '.c',"w") as f:
        #         f.write(str(data[i]))
        #
        # print("Code process done!")
        # cur_order = r'java -jar cli.jar pathContexts --lang c --project ' + self.predict_code_path + ' --output ' + self.parsed_code_path + \
        #             r' --maxH 8 --maxW 2 --maxContexts ' + str(self.config.MAX_CONTEXTS) + ' --maxTokens ' + str(
        #     self.config.MAX_TOKEN_VOCAB_SIZE) + \
        #             ' --maxPaths ' + str(self.config.MAX_PATH_VOCAB_SIZE)
        # ret = os.system(cur_order)
        # assert ret == 0

        node_types_path = os.path.join(self.parsed_code_path, "node_types.csv")
        paths_path = os.path.join(self.parsed_code_path, "paths.csv")
        tokens_path = os.path.join(self.parsed_code_path, "tokens.csv")

        file1 = pd.read_csv(node_types_path)
        file2 = pd.read_csv(paths_path)
        file3 = pd.read_csv(tokens_path)

        temp_node_types_dict = {}
        temp_paths_dict = {}
        temp_tokens_dict = {}

        def node_types2dict(row):
            temp_node_types_dict[row['id']] = row['node_type']

        file1.apply(node_types2dict, axis=1)

        def paths2dict(row):
            temp_paths_dict[row['id']] = row['path']

        file2.apply(paths2dict, axis=1)

        def tokens2dict(row):
            temp_tokens_dict[row['id']] = row['token']

        file3.apply(tokens2dict, axis=1)

        source_inputs =[]
        path_inputs = []
        target_inputs = []
        context_valid_inputs = []
        map1 = {}
        cnt2 = 0
        for i in range(0,160):
            code_path = os.path.join(self.parsed_code_path, "path_contexts_" + str(i) + ".csv")
            file4 = open(code_path, "r")
            while(True):
                source_input = []
                path_input = []
                target_input = []
                context_valid_input = []

                codeline = file4.readline()
                if codeline:
                    pass
                else:
                    break
                code = codeline.split(' ')[1:]
                name = codeline.split(' ')[0].split('.')[-2].split('/')[-1]
                map1[name] = cnt2
                cnt2=cnt2+1
                cnt = 0

                for path in code:
                    temp_source, temp_path, temp_target = map(int, path.split(','))
                    try:
                        real_source = self.tokens_dict[temp_tokens_dict[temp_source]]
                        real_target = self.tokens_dict[temp_tokens_dict[temp_target]]
                        real_path = self.paths_dict[' '.join(list(map(lambda x: str(self.node_types_dict[x]),
                                                                      list(map(lambda x: temp_node_types_dict[int(x)],
                                                                               temp_paths_dict[temp_path].split(' '))))))]
                        context_valid_input.append(1)
                    except:
                        # if the path cannot map into the vocabulary due to dataset reason or unknown cli.jar technical stargety
                        # I currently choose to set value to zero to delete the path
                        real_path = 0
                        context_valid_input.append(0)
                        real_source = 0
                        real_target = 0

                    source_input.append(real_source)
                    path_input.append(real_path)
                    target_input.append(real_target)
                    # print(real_source,real_path,real_target)
                    cnt = cnt + 1

                source_input = np.pad(source_input, (0, self.config.MAX_CONTEXTS - cnt), 'constant', constant_values=(0, 0))
                path_input = np.pad(path_input, (0, self.config.MAX_CONTEXTS - cnt), 'constant', constant_values=(0, 0))
                target_input = np.pad(target_input, (0, self.config.MAX_CONTEXTS - cnt), 'constant', constant_values=(0, 0))
                context_valid_input = np.pad(context_valid_input, (0, self.config.MAX_CONTEXTS - cnt), 'constant', constant_values=(0, 0))

                source_inputs.append(source_input)
                path_inputs.append(path_input)
                target_inputs.append(target_input)
                context_valid_inputs.append(context_valid_input)

            file4.close()

        inputs = (source_inputs,path_inputs,target_inputs,context_valid_inputs)
        results = temp_model.predict(inputs)

        final = [[] for i in range(192)]
        cur_csv = pd.read_csv(code_csv_path, encoding='gbk')
        for i in range(len(cur_csv)):
            cur_id = cur_csv.iloc[i]['w']
            if (str(cur_id) == 'nan'):
                for j in range(192):
                    final[j].append(None)
                continue
            for j in range(192):
                cur_result = results[map1[cur_id]]
                final[j].append(cur_result[j])

        for j in range(192):
            cur_csv['codeDim_' + str(j)] = final[j]

        cur_csv.to_csv('./submission1.csv',index=False)


if __name__ == "__main__":
    os.chdir(sys.path[0])

    model = Code2tags(problem_info = False)

    # load model
    modelSaved_path = os.path.join(model.modelSaved_path, "origin_model")
    model.load_savedModel(os.path.join(modelSaved_path, "cp-0080.ckpt"))
    code = "#include <iostream>" \
           "int main() {" \
           "int a, b;" \
           "std::cin >> a >> b;" \
           "std::cout << a + b;" \
           "return 0;" \
           "}"
    res = model.predict_code(code)
    print(res)

    # data = model.load_data_2()
    # model.train(data[0])
    # res = model.evaluate(data[1][:-2],data[1][-2:])
    # print(res)