from utils.api import APIView
from account.decorators import login_required
import numpy as np
from problem.models import Problem
from .admin import SIMILAR_MATRIX, STAR_MATRIX, USER_NAME, USER_NEW, NEW_USER, N_D

class RecommendAPI(APIView):
    def init(self):
        try:
            self.similar_matrix = np.load(SIMILAR_MATRIX)
            self.n_M_star = np.load(STAR_MATRIX)
            self.user2new_map = np.load(USER_NEW, allow_pickle=True).item()
            self.new2user_map = np.load(NEW_USER, allow_pickle=True).item()
            self.n_D_map = np.load(N_D, allow_pickle=True).item()
            self.user2name_map = np.load(USER_NAME, allow_pickle=True).item()
            self.problem_number = self.n_M_star.shape[1]
            return True
        except:
            return False

    @login_required
    def get(self, request):
        if self.init() == False:
            return self.error("文件加载错误")

        try:
            user_id = int(request.GET.get('user_id'))
            user_number = int(request.GET.get('user_number'))
            exercise_number = int(request.GET.get('exercise_number'))
        except:
            return self.error("参数错误")

        # 查询系统最活跃的k个用户
        def get_top_k_active_user(k, D_map):
            sub_map = {}
            for key in D_map.keys():
                uid = int(key.split('-')[0])
                if uid not in sub_map.keys():
                    sub_map[uid] = 0
                sub_map[uid] += 1
            sort_map = sorted(sub_map.items(), key=lambda x: x[1], reverse=True)
            idx = 0
            active_user = []
            for uid, times in sort_map:
                if idx == k:
                    break
                idx += 1
                active_user.append(uid)
            return active_user

        def get_tok_k_similar_user(k, similar_M, x, user2index_map, index2user_map, D_map):
            user_num, _ = similar_M.shape
            new_user = False
            if x not in user2index_map.keys():
                # print('新用户，推荐k个活跃用户')
                new_user = True
                nf_user_ids = get_top_k_active_user(k, D_map)
            else:
                n_uid = user2index_map[x]
                similars = similar_M[n_uid]
                sort_indexs = np.argsort(-similars)
                n_user_ids = sort_indexs[:k]
                nf_user_ids = []
                for idx, uid in enumerate(n_user_ids):
                    if uid != n_uid:
                        nf_user_ids.append(uid)
                if len(nf_user_ids) < k:
                    nf_user_ids.append(sort_indexs[k])
            row_user_ids = [index2user_map[i] for i in nf_user_ids]
            return nf_user_ids, row_user_ids, new_user

        n_ids, row_ids, new_user = get_tok_k_similar_user(user_number,self.similar_matrix,user_id,self.user2new_map,self.new2user_map,self.n_D_map)

        cur_users = [self.user2name_map[i] for i in row_ids]

        def get_recommend_exercise(k1, k2, similar_M, x, user2index_map, index2user_map, D_map, pro_num, M_star):
            scores_p = np.zeros(pro_num)
            n_ids, row_ids, new_user = get_tok_k_similar_user(k1, self.similar_matrix, x, self.user2new_map, self.new2user_map,
                                                              self.n_D_map)

            for p_idx in range(pro_num):
                for n_uid in n_ids:
                    sim = 1
                    if not new_user:
                        sim = similar_M[user2index_map[x]][n_uid]
                    scores_p[p_idx] += sim * M_star[n_uid][p_idx]

            return np.argsort(-scores_p)[:k2]

        p_idxs = get_recommend_exercise(user_number,exercise_number , self.similar_matrix, user_id, self.user2new_map, self.new2user_map, self.n_D_map, self.problem_number,
                                        self.n_M_star)

        p_idxs = [int(id + 1) for id in p_idxs]

        p_infos = list(Problem.objects.filter(id__in= p_idxs).values('_id', 'title','contest_id'))

        return self.success({'user': cur_users, 'problems': p_idxs, 'problems_info': p_infos})