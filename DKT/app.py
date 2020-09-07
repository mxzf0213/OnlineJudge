from utils.api import APIView
import numpy as np
from DKT.dkt_utils import *
import DKT.dkt_utils as utils
import torch
from account.decorators import login_required
from submission.models import Submission
from problem.models import Problem
from django.db.models import Q
import sys

sys.path.append('DKT')

saved_path = 'dataset1'
dkt_model = torch.load('DKT/save/' + saved_path)
seq_len = 200

hw_contestIds = [20,21,22,23,24,26,28,30,31,34]
knowledge_ids = [29, 68, 64, 69, 15, 62, 59, 48, 67, 63, 25, 26, 58, 65, 66, 72, 73, 18, 44, 70, 71]
knowledge_names = ['数论','循环语句','条件语句','表达式','数组','格式化输入输出','函数','递归','基本数据类型','结构体','排序',
                   '字符串','指针','暴力','动态规划','结构体数组','判断语句','链表','递推','动态分配','预处理']

class DktAPI(APIView):
    def predict(self,dkt_model, x_data):
        batch_size = x_data.shape[0]
        seqlen = x_data.shape[1]

        init_h = variable(torch.randn(dkt_model.hidden_layers, batch_size, dkt_model.hidden_dim), -1)
        init_c = variable(torch.randn(dkt_model.hidden_layers, batch_size, dkt_model.hidden_dim), -1)

        # modify
        x_data = x_data.view(batch_size * seqlen, -1)
        x_data = dkt_model.x_linear(x_data)
        x_data = x_data.view(batch_size, seqlen, -1)

        ## lstm process
        lstm_out, final_status = dkt_model.rnn(x_data, (init_h, init_c))

        prediction = dkt_model.predict_linear(lstm_out.view(batch_size * seqlen, -1))

        pred_out = torch.tanh(prediction)

        return pred_out[-1]

    @login_required
    def get(self, request):
        user = request.user
        problem_id = request.GET.get('problem_id')
        cur_tags = Problem.objects.all().filter(id = problem_id).values('tags')
        submissions = list(Submission.objects.all().filter(Q(user_id = user.id) & Q(contest_id__in = hw_contestIds)).\
            order_by('-create_time')[:seq_len].values('problem_id', 'info'))

        x_data = np.zeros((1,seq_len,dkt_model.n_question * 4 + 1))

        dif = seq_len - len(submissions)

        for i in range(len(submissions)):
            info = submissions[i]['info']
            if info != {}:
                data_dic = info['data']
                pass_prob = sum(list(map(lambda x:x['result'] == 0, data_dic))) / len(data_dic)
            else:
                pass_prob = 0

            problem_tags = Problem.objects.all().filter(id = submissions[i]['problem_id']).values('tags')
            for x in problem_tags:
                problem_tag = x['tags']
                if pass_prob == 0:
                    x_index = problem_tag
                elif pass_prob == 1.0:
                    x_index = problem_tag + int(pass_prob * 4.0 - 1) * dkt_model.n_question
                else:
                    x_index = problem_tag + int(np.floor(pass_prob * 4.0 - 1e-4)) * dkt_model.n_question

                x_data[0, dif + i, x_index] = 1

        input_x = utils.variable(torch.FloatTensor(x_data), -1)
        res = self.predict(dkt_model, input_x)[knowledge_ids]
        return self.success({'res' : res.tolist(), 'name': knowledge_names})


# if __name__ == "__main__":
#     x_data = np.ones((1, 200, dkt_model.n_question * 4 + 1))
#     input_x = utils.variable(torch.FloatTensor(x_data), -1)
#     res = predict(dkt_model, input_x)
#     print(res.shape)
