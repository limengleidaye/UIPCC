import pandas as pd
from util import DataSet
from tqdm import tqdm
import numpy as np


class UPICC():
    # 初始化相关参数
    def __init__(self):
        # 找到与目标用户兴趣相似的20个用户，为其推荐10部电影
        self.n_sim_user = 100

        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}
        self.predict_metrix_U = []
        self.predict_metrix_S = []

        # 用户相似度矩阵
        self.user_sim_matrix = {}
        self.movie_sim_matrix = {}
        self.movie_count = 0
        self.user_count = 0

    # 读文件得到“用户-电影”数据
    def get_dataset(self, filename):
        feature, train, test = DataSet(file=filename).create_explicit_ml_1m_dataset()
        self.user_count = feature[0]['feat_num']
        self.movie_count = feature[1]['feat_num']
        self.predict_metrix_U = np.zeros(shape=(self.user_count + 1, self.movie_count + 1))
        self.predict_metrix_S = np.zeros(shape=(self.user_count + 1, self.movie_count + 1))

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

    # 计算用户之间的相似度
    def calc_user_sim(self):
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

        print('Calculating user predict ratings...')
        for user in tqdm(range(1, self.user_count + 1)):
            for movie in self.trainSet[user].keys():
                self.predict_metrix_U[user][movie] = self.trainSet[user][movie]
            temp_df = pd.DataFrame(columns=['MovieId', 'sum_rating', 'sum_sim']).set_index('MovieId')
            # 找相似邻居
            temp_rank = sorted(self.user_sim_matrix[user].items(), key=lambda x: x[1],
                               reverse=True)[:40]
            for nei, sim in temp_rank:
                if self.user_sim_matrix[user][nei]>0.009:
                    for movie in self.trainSet[nei].keys():
                        # 如果相似邻居看过这部电影并且自己没看过
                        if movie not in self.trainSet[user].keys():
                            # 判断movie是否在rank中
                            if movie in temp_df.index.values:
                                temp_df.loc[movie] = temp_df.loc[movie].values + [
                                    self.user_sim_matrix[user][nei] * self.trainSet[nei][movie],
                                    self.user_sim_matrix[user][nei]]
                            else:
                                temp_df.loc[movie] = [self.user_sim_matrix[user][nei] * self.trainSet[nei][movie],
                                                      self.user_sim_matrix[user][nei]]
            temp_df['rating'] = temp_df['sum_rating'] / temp_df['sum_sim']
            for movie in temp_df.index.values:
                self.predict_metrix_U[user][movie] = temp_df.loc[movie]['rating']
        pd.DataFrame(self.predict_metrix_U, columns=range(self.movie_count + 1)).to_csv(
            './predict_metrix_U.csv', index=False)
        print('Calculate user predict ratings success!')

    # 计算电影之间的相似度
    def calc_movie_sim(self):
        # key = movieID, value = list of userIDs who have seen this movie
        print('Building movie-user table ...')
        movie_user = {}
        for user, movies in self.trainSet.items():
            for movie in movies:
                if movie not in movie_user:
                    movie_user[movie] = set()
                movie_user[movie].add(user)
        print('Build movie-user table success!')

        print('Building movie sim table ...')
        for s, s_users in tqdm(movie_user.items()):
            for g, g_users in movie_user.items():
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

        print('Calculating movie predict matrix...')
        for movie in tqdm(movie_user.keys()):
            for user in movie_user[movie]:
                self.predict_metrix_U[user][movie] = self.trainSet[user][movie]
            temp_df = pd.DataFrame(columns=['UserId', 'sum_rating', 'sum_sim']).set_index('UserId')
            # 找相似邻居
            temp_rank = sorted(self.movie_sim_matrix[movie].items(), key=lambda x: x[1],
                               reverse=True)[:40]
            for nei, sim in temp_rank:
                if self.movie_sim_matrix[movie][nei]>0.009:
                    for user in movie_user[nei]:
                        # 如果相似邻居看过这部电影并且自己没看过
                        if user not in movie_user[movie]:
                            # 判断movie是否在rank中
                            if user in temp_df.index.values:
                                temp_df.loc[user] = temp_df.loc[user].values + [
                                    self.movie_sim_matrix[movie][nei] * self.trainSet[user][nei],
                                    self.movie_sim_matrix[movie][nei]]
                            else:
                                temp_df.loc[user] = [self.movie_sim_matrix[movie][nei] * self.trainSet[user][nei],
                                                     self.movie_sim_matrix[movie][nei]]
            temp_df['rating'] = temp_df['sum_rating'] / temp_df['sum_sim']
            for user in temp_df.index.values:
                self.predict_metrix_S[user][movie] = temp_df.loc[user]['rating']
        pd.DataFrame(self.predict_metrix_S, columns=range(self.movie_count + 1)).to_csv(
            './predict_metrix_S.csv', index=False)
        print('Calculate movie predict matrix success!')


if __name__ == '__main__':
    rating_file = './data/ml-1m/ratings.dat'
    UPICC = UPICC()
    UPICC.get_dataset(rating_file)
    UPICC.calc_movie_sim()
    UPICC.calc_user_sim()
    # userCF.evaluate()
