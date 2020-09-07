from utils.api import APIView
from account.decorators import super_admin_required
from submission.models import Submission
import pandas as pd
import numpy as np
from time import strftime, localtime
import os,sys

# USER_HOME = os.path.expanduser('~')
USER_HOME = 'recommend'
SIMILAR_MATRIX = os.path.join(USER_HOME, 'similar_matrix.npy')
STAR_MATRIX = os.path.join(USER_HOME,'star_matrix.npy')
USER_NEW = os.path.join(USER_HOME,'user_new.npy')
NEW_USER = os.path.join(USER_HOME,'new_user.npy')
USER_NAME = os.path.join(USER_HOME,'user_name.npy')
N_D = os.path.join(USER_HOME,'n_D.npy')
PROBLEM_NUMBER = 0

def get():
    submission_data = pd.DataFrame(Submission.objects.all().order_by('create_time').values())
    PROBLEM_NUMBER = max(set(submission_data['problem_id'])) + 1

    block_user_ids = [
        #     1,280,358,275-
    ]

    def get_D_map(submission_data):
        D_map = {}
        for index, row in submission_data.iterrows():
            pid = row['problem_id']
            uid = row['user_id']
            create_time = row['create_time']
            if uid in block_user_ids:
                continue
            key = str(uid) + '-' + str(pid)
            username = row['username']
            if key not in D_map.keys():
                D_map[key] = (0, 0)
            try_times, solved = D_map[key]
            #     if solved:
            #         print(f'用户 {uid}-{username} 在做对题目{pid}之后重复尝试 提交时间{create_time}')
            #         continue
            if int(row['result']) == 0:
                solved = 1
            #         print(f'用户 {uid}-{username} 题目{pid} AC 提交时间{create_time}')
            try_times += 1
            D_map[key] = (try_times, solved)
        return D_map

    def re_index_k_map(D_map):
        user2new_map = {}
        new2user_map = {}
        n_D_map = {}
        index = 0
        for key in D_map.keys():
            uid = int(key.split('-')[0])
            pid = int(key.split('-')[1])
            if uid not in user2new_map.keys():
                user2new_map[uid] = index
                new2user_map[index] = uid
                index += 1
            n_uid = user2new_map[uid]
            n_key = str(n_uid) + '-' + str(pid)
            n_D_map[n_key] = D_map[key]
        return n_D_map, user2new_map, new2user_map

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

    D_map = get_D_map(submission_data)

    n_D_map, user2new_map, new2user_map = re_index_k_map(D_map)

    def get_NS_S(D_map, pro_num):
        Ns = np.zeros(pro_num)
        S = np.zeros(pro_num)
        s_times = np.zeros(pro_num)
        ns_times = np.zeros(pro_num)
        for key in D_map.keys():
            uid = int(key.split('-')[0])
            pid = int(key.split('-')[1])
            try_times, solved = D_map[key]
            if solved:
                S[pid - 1] += try_times
                s_times[pid - 1] += 1
            else:
                Ns[pid - 1] += try_times
                ns_times[pid - 1] += 1
        lambda_p1 = Ns / ns_times
        lambda_p2 = S / s_times
        lambda_p1[np.isnan(lambda_p1)] = 0
        lambda_p2[np.isnan(lambda_p2)] = 0
        return np.round(lambda_p1), np.round(lambda_p2)

    lambda_p1, lambda_p2 = get_NS_S(n_D_map, PROBLEM_NUMBER)

    def get_M_star(D_map, user_num, pro_num, lambda_p1, lambda_p2):
        M = np.zeros((user_num, pro_num))
        for uid in range(len(M)):
            for p_idx in range(len(M[0])):
                pid = p_idx + 1
                key = str(uid) + '-' + str(pid)
                if key in D_map.keys():
                    try_times, solved = D_map[key]
                    if not solved:
                        if try_times >= lambda_p1[pid - 1]:
                            M[uid][p_idx] = 1
                        else:
                            M[uid][p_idx] = 2
                    else:
                        if try_times >= lambda_p2[pid - 1]:
                            M[uid][p_idx] = 3
                        else:
                            M[uid][p_idx] = 4
        return M

    M_star = get_M_star(n_D_map, len(user2new_map), PROBLEM_NUMBER, lambda_p1, lambda_p2)

    def get_C1_C2(M_star):
        C1u_map = {}
        C2u_map = {}
        C1p_map = {}
        C2p_map = {}
        for uid in range(len(M_star)):
            for p_idx in range(len(M_star[0])):
                pid = p_idx + 1
                if uid not in C1u_map.keys():
                    C1u_map[uid] = set()
                if uid not in C2u_map.keys():
                    C2u_map[uid] = set()
                if pid not in C1p_map.keys():
                    C1p_map[pid] = set()
                if pid not in C2p_map.keys():
                    C2p_map[pid] = set()
                if M_star[uid][p_idx] == 3:
                    C1u_map[uid].add(pid)
                    C1p_map[pid].add(uid)
                if M_star[uid][p_idx] == 4:
                    C2u_map[uid].add(pid)
                    C2p_map[pid].add(uid)
        return C1u_map, C2u_map, C1p_map, C2p_map

    C1u_map, C2u_map, C1p_map, C2p_map = get_C1_C2(M_star)

    def correct_M_star(M_star, C1u_map, C2u_map, C1p_map, C2p_map):
        for uid in range(len(M_star)):
            for p_idx in range(len(M_star[0])):
                pid = p_idx + 1
                if M_star[uid][p_idx] == 3:
                    if (len(C2u_map[uid]) > 0 or (len(C1u_map[uid]) > 0)) and (
                            len(C2p_map[pid]) > 0 or (len(C1p_map[pid]) > 0)) and (
                            len(C2u_map[uid]) >= 2 * len(C1u_map[uid])) and (
                            len(C2p_map[pid]) >= 2 * len(C1p_map[pid])):
                        M_star[uid][p_idx] = 4
                    if M_star[uid][p_idx] == 4:
                        if (len(C2u_map[uid]) > 0 or (len(C1u_map[uid]) > 0)) and (
                                len(C2p_map[pid]) > 0 or (len(C1p_map[pid]) > 0)) and (
                                len(C1u_map[uid]) >= 2 * len(C2u_map[uid])) and (
                                len(C1p_map[pid]) >= 2 * len(C2p_map[pid])):
                            M_star[uid][p_idx] = 3
        return M_star

    n_M_star = correct_M_star(M_star, C1u_map, C2u_map, C1p_map, C2p_map)

    def get_similar_matrix(M_star):
        user_num, pro_num = M_star.shape
        similar_M = np.zeros((user_num, user_num))
        for u1 in range(len(similar_M)):
            for u2 in range(len(similar_M[0])):
                Sxu = 0
                Dxu = 0
                for p_idx in range(pro_num):
                    if M_star[u1][p_idx] == M_star[u2][p_idx] and M_star[u1][p_idx] > 0:
                        Sxu += 1
                    if M_star[u2][p_idx] > 0 and M_star[u1][p_idx] > 0:
                        Dxu += 1
                if Dxu > 0:
                    similar_M[u1][u2] = Sxu / Dxu
        return similar_M

    similar_matrix = get_similar_matrix(n_M_star)

    def get_tok_k_similar_user(k, similar_M, x, user2index_map, index2user_map, D_map):
        user_num, _ = similar_M.shape
        n_user_ids = []
        new_user = False
        if x not in user2index_map.keys():
            # print('新用户，推荐k个活跃用户')
            new_user = True
            n_user_ids = get_top_k_active_user(k, D_map)
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

    def get_userid2name_map(submission_data):
        id2name_map = {}
        for index, row in submission_data.iterrows():
            uid = row['user_id']
            name = row['username']
            if uid not in id2name_map.keys():
                id2name_map[uid] = name
        return id2name_map

    user2name_map = get_userid2name_map(submission_data)

    np.save(SIMILAR_MATRIX, similar_matrix)
    np.save(STAR_MATRIX, n_M_star)

    np.save(USER_NEW, user2new_map)
    np.save(NEW_USER, new2user_map)
    np.save(N_D, n_D_map)
    np.save(USER_NAME, user2name_map)

    print("recommend update success!")
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))





