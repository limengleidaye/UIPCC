import pandas as pd
from util import DataSet
from tqdm import tqdm
import numpy as np


class UIPCC():
    def __init__(self):
        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}
        self.movie_user = {}  # 电影 用户倒排表
        self.user_avg_std = {}

        # 用户相似度矩阵
        self.user_sim_matrix = {}
        self.movie_sim_matrix = {}
        self.movie_count = 0
        self.user_count = 0

        # ==================Init=======================================
        self.UPCC = UPCC()
        self.IPCC = IPCC()
        self.UPCC.trainSet = self.trainSet
        self.UPCC.testSet = self.testSet
        self.UPCC.user_avg_std = self.user_avg_std
        self.UPCC.movie_user = self.movie_user
        self.UPCC.movie_count = self.movie_count
        self.UPCC.user_count = self.user_count
        self.IPCC.trainSet = self.trainSet
        self.IPCC.testSet = self.testSet
        self.IPCC.user_avg_std = self.user_avg_std
        self.IPCC.movie_user = self.movie_user
        self.IPCC.movie_count = self.movie_count
        self.IPCC.user_count = self.user_count

        # =================得到数据集，计算相似度================================
        self.get_dataset()
        self.cal_sim()

    def get_dataset(self):
        dataset = DataSet()
        feature, train, test = dataset.create_explicit_ml_1m_dataset()
        self.user_count = feature[0]['feat_num']
        self.movie_count = feature[1]['feat_num']

        for line in train:
            user, movie, rating = line
            self.trainSet.setdefault(user.astype('int'), {})
            self.trainSet[user.astype('int')][movie.astype('int')] = rating
        for line in test:
            user, movie, rating = line
            self.testSet.setdefault(user.astype('int'), {})
            self.testSet[user.astype('int')][movie.astype('int')] = rating
        print('Split trainingSet and testSet success!')
        print('TrainSet = %s' % len(train))
        print('TestSet = %s' % len(test))
        print('Building movie-user table ...')
        for user, movies in self.trainSet.items():
            for movie in movies:
                if movie not in self.movie_user:
                    self.movie_user[movie] = set()
                self.movie_user[movie].add(user)
        print('Build movie-user table success!')

        self.user_avg_std = dataset.get_mean_std().set_index('UserId').to_dict('index')

    def cal_sim(self):
        self.UPCC.cal_user_sim()
        self.IPCC.cal_movie_sim()

    def predict(self, user, list):
        upcc_predict = self.UPCC.predict(user, list)
        ipcc_predict = self.IPCC.predict(user, list)
        predict_dict = {}
        for movie in upcc_predict:
            predict_dict[movie] = upcc_predict[movie] * 0.9 + ipcc_predict[movie] * 0.1
        return predict_dict

    def evaluate(self, list):
        upcc_eva = self.UPCC.evaluate(list)
        ipcc_eva = self.IPCC.evaluate(list)
        predict_list = []
        for idx, item in enumerate(upcc_eva):
            predict_list.append([item[0], upcc_eva[idx][1] * 0.9 + ipcc_eva[idx][1] * 0.1])
        return predict_list


class IPCC():
    def __init__(self):
        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}
        self.movie_user = {}  # 电影 用户倒排表
        self.user_avg_std = {}

        # 用户相似度矩阵
        self.movie_sim_matrix = {}
        self.movie_count = 0
        self.user_count = 0

        # =================得到数据集，计算相似度================================
        # self.get_dataset()
        self.cal_movie_sim()

    # 读文件得到“用户-电影”数据
    def get_dataset(self):
        dataset = DataSet()
        feature, train, test = dataset.create_explicit_ml_1m_dataset()
        self.user_count = feature[0]['feat_num']
        self.movie_count = feature[1]['feat_num']

        for line in train:
            user, movie, rating = line
            self.trainSet.setdefault(user.astype('int'), {})
            self.trainSet[user.astype('int')][movie.astype('int')] = rating
        for line in test:
            user, movie, rating = line
            self.testSet.setdefault(user.astype('int'), {})
            self.testSet[user.astype('int')][movie.astype('int')] = rating
        print('Split trainingSet and testSet success!')
        print('TrainSet = %s' % len(train))
        print('TestSet = %s' % len(test))
        print('Building movie-user table ...')
        for user, movies in self.trainSet.items():
            for movie in movies:
                if movie not in self.movie_user:
                    self.movie_user[movie] = set()
                self.movie_user[movie].add(user)
        print('Build movie-user table success!')

        self.user_avg_std = dataset.get_mean_std().set_index('UserId').to_dict('index')

    # 计算电影之间的相似度
    def cal_movie_sim(self):
        # key = movieID, value = list of userIDs who have seen this movie
        print('Building movie sim table ...')
        for s, s_users in tqdm(self.movie_user.items()):
            for g, g_users in self.movie_user.items():
                if s == g:
                    continue
                same_users = s_users & g_users
                if len(same_users) != 0:
                    sum_numerator = sum(self.trainSet[user][s] * self.trainSet[user][g] for user in same_users)
                    sum_denominator = np.sqrt(len(s_users) * len(g_users))
                    self.movie_sim_matrix.setdefault(s, {})
                    self.movie_sim_matrix[s].setdefault(g, 0)
                    self.movie_sim_matrix[s][g] = sum_numerator / sum_denominator
        print('Build user similarity matrix success!')

    def predict(self, user, list):
        predict_dict = {}
        for movie in list:
            predict_dict[movie] = 0
            try:
                movies = self.trainSet[user].keys()
            except Exception:
                # print('no one has watched the movie ',movie)
                continue
            temp_sum_upper = 0  # 分子
            temp_sum_down = 0  # 分母
            # simest_neis = [item[0] for item in
            #                sorted(self.movie_sim_matrix[user].items(), key=lambda x: x[1], reverse=True)[:80]]
            for nei in movies:
                try:
                    if nei in self.movie_sim_matrix[movie].keys():
                        temp_sum_upper += self.movie_sim_matrix[movie][nei] * self.trainSet[user][nei]
                        temp_sum_down += self.movie_sim_matrix[movie][nei]
                except Exception:
                    pass
            if temp_sum_down != 0:
                predict_dict[movie] = temp_sum_upper / temp_sum_down
        return predict_dict

    def evaluate(self, list):
        predict_list = []
        for [user, movie] in list:
            try:
                movies = self.trainSet[user].keys()
            except Exception:
                # print('no one has watched the movie ',movie)
                predict_list.append([user, 0])
                continue
            temp_sum_upper = 0  # 分子
            temp_sum_down = 0  # 分母
            for nei in movies:
                try:
                    if nei in self.movie_sim_matrix[movie].keys():
                        temp_sum_upper += self.movie_sim_matrix[movie][nei] * self.trainSet[user][nei]
                        temp_sum_down += self.movie_sim_matrix[movie][nei]
                except Exception:
                    pass
            if temp_sum_down != 0:
                predict_list.append([user, np.clip(temp_sum_upper / temp_sum_down, 0.5, 5.5)])
            else:
                predict_list.append([user, 0])
        return predict_list


class UPCC():
    # 初始化相关参数
    def __init__(self):
        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}
        self.movie_user = {}  # 电影 用户倒排表
        self.user_avg_std = {}

        # 用户相似度矩阵
        self.user_sim_matrix = {}
        self.movie_count = 0
        self.user_count = 0

        # =================得到数据集，计算相似度================================
        self.get_dataset()
        self.cal_user_sim()

    # 读文件得到“用户-电影”数据
    def get_dataset(self):
        dataset = DataSet()
        feature, train, test = dataset.create_explicit_ml_1m_dataset()
        self.user_count = feature[0]['feat_num']
        self.movie_count = feature[1]['feat_num']

        for line in train:
            user, movie, rating = line
            self.trainSet.setdefault(user.astype('int'), {})
            self.trainSet[user.astype('int')][movie.astype('int')] = rating
        for line in test:
            user, movie, rating = line
            self.testSet.setdefault(user.astype('int'), {})
            self.testSet[user.astype('int')][movie.astype('int')] = rating
        print('Split trainingSet and testSet success!')
        print('TrainSet = %s' % len(train))
        print('TestSet = %s' % len(test))
        print('Building movie-user table ...')
        for user, movies in self.trainSet.items():
            for movie in movies:
                if movie not in self.movie_user:
                    self.movie_user[movie] = set()
                self.movie_user[movie].add(user)
        print('Build movie-user table success!')

        self.user_avg_std = dataset.get_mean_std().set_index('UserId').to_dict('index')

    # 计算用户之间的相似度
    def cal_user_sim(self):
        # key = movieID, value = list of userIDs who have seen this movie
        print('Building user sim table ...')
        for u, u_movies in tqdm(self.trainSet.items()):
            for v, v_movies in self.trainSet.items():
                if u == v:
                    continue
                same_movies = set(u_movies.keys()) & set(v_movies.keys())
                if len(same_movies) != 0:
                    sum_numerator = sum(self.trainSet[u][movie] * self.trainSet[v][movie] for movie in same_movies)
                    sum_denominator = np.sqrt(len(u_movies) * len(v_movies))
                    self.user_sim_matrix.setdefault(u, {})
                    self.user_sim_matrix[u].setdefault(v, 0)
                    self.user_sim_matrix[u][v] = sum_numerator / sum_denominator
        print('Calculate user similarity matrix success!')

    def predict(self, user, list):
        predict_dict = {}
        for movie in list:
            predict_dict[movie] = 0
            try:
                users = self.movie_user[movie]
            except Exception:
                # print('no one has watched the movie ',movie)
                continue
            temp_sum_upper = 0  # 分子
            temp_sum_down = 0  # 分母
            # simest_neis = [item[0] for item in
            #                sorted(self.user_sim_matrix[user].items(), key=lambda x: x[1], reverse=True)[:80]]
            for nei in users:
                if nei in self.user_sim_matrix[user].keys():
                    temp_sum_upper += self.user_sim_matrix[user][nei] * self.trainSet[nei][movie]
                    temp_sum_down += self.user_sim_matrix[user][nei]
            if temp_sum_down != 0:
                predict_dict[movie] = temp_sum_upper / temp_sum_down
        return predict_dict

    def evaluate(self, list):
        predict_list = []
        for [user, movie] in list:
            try:
                users = self.movie_user[movie]
            except Exception:
                # print('no one has watched the movie ',movie)
                predict_list.append([user, 0])
                continue
            temp_sum_upper = 0  # 分子
            temp_sum_down = 0  # 分母
            for nei in users:
                if nei in self.user_sim_matrix[user].keys():
                    temp_sum_upper += self.user_sim_matrix[user][nei] * self.trainSet[nei][movie]
                    temp_sum_down += self.user_sim_matrix[user][nei]
            if temp_sum_down != 0:
                predict_list.append([user, temp_sum_upper / temp_sum_down])
            else:
                predict_list.append([user, 0])
        return predict_list
