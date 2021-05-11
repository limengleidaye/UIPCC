import pandas as pd
import numpy as np
from util import DataSet
from tqdm import tqdm
import random


class Evaluate:
    def __init__(self):
        u_data_df = pd.read_csv('./predict_metrix_U.csv', engine='python')
        s_data_df = pd.read_csv('./predict_metrix_S.csv', engine='python').fillna(value=0.0)
        data_df = u_data_df.drop(index=0, columns='0').stack().reset_index()
        data_df.columns = ['UserId', 'MovieId', 'Rating']

        # ==================获取训练集 测试集========================
        dataset = DataSet('./data/ml-1m/ratings.dat')
        feature_column, train, test = dataset.create_explicit_ml_1m_dataset()
        item_num = feature_column[1]['feat_num']
        self.all_items = set(np.arange(1, item_num + 1))
        self.train_df = pd.DataFrame(train, columns=['UserId', 'MovieId', 'Rating']).set_index('UserId')
        self.test_df = pd.DataFrame(test, columns=['UserId', 'MovieId', 'Rating']).set_index('UserId')
        # ====================获取用户方差，均值=====================
        mean_std_data_df = dataset.get_mean_std()
        data = np.multiply(data_df['Rating'],np.repeat(mean_std_data_df['user_var'].values, feature_column[1]['feat_num'])) + np.repeat(
            mean_std_data_df['user_avg'].values, feature_column[1]['feat_num'])
        # ======================建立预测评分=========================
        self.predict_rating = pd.DataFrame(
            np.transpose(
                [data_df['UserId'].values.astype('int64'), data_df['MovieId'].values.astype('int64'), data]),
            columns=['UserId', 'MovieId', 'Rating']).set_index(['UserId', 'MovieId'])
        return

    def recommend(self, user, list, n):
        return self.predict_rating.loc[user].loc[list].sort_values(by='Rating', ascending=False)[:n].index.values

    def rec(self, N):
        print("Evaluation start ...", N)
        pre_list = []
        # recall_list = []
        # 准确率和召回率
        test = self.test_df[self.test_df['Rating'] > 4]
        for n in N:
            # rec_count = 0
            test_count = 0
            hit = 0
            for user in tqdm(test.index.unique()):
                train_items = set(self.train_df.loc[user]['MovieId'].values.astype('int64'))
                try:
                    test_items = set(test.loc[user]['MovieId'].values.astype('int64'))
                except Exception:
                    test_items = set([int(test.loc[user]['MovieId'])])
                other_items = self.all_items - train_items.union(test_items)
                for idx in test_items:
                    random_items = random.sample(other_items, 200)
                    random_items.append(idx)
                    rec_movies = self.recommend(user, random_items, n).astype('int64')
                    hit += int(idx in rec_movies)
                    # rec_count += n
                    test_count += len(test_items)

            # precision = hit / (1.0 * rec_count)
            recall = hit / (1.0 * test_count)
            # pre_list.append(precision)
            # recall_list.append(recall)
            print('N:%d\trecall=%.10f\t' % (n, recall))

    def rmse(self):
        test_data = self.test_df.reset_index()[['UserId', 'MovieId']].values.astype('int64').tolist()
        pre_rating = self.predict_rating.loc[[tuple(x) for x in test_data]]['Rating'].values
        print('rmse : ', np.sqrt(np.sum(np.power(pre_rating - self.test_df['Rating'].values, 2)) / len(pre_rating)))

    def mae(self):
        test_data = self.test_df.reset_index()[['UserId', 'MovieId']].values.astype('int64').tolist()
        pre_rating = self.predict_rating.loc[[tuple(x) for x in test_data]]['Rating'].values
        print('mae: ', np.sum(np.abs(pre_rating - self.test_df['Rating'].values)) / len(pre_rating))


if __name__ == '__main__':
    test = Evaluate()
    test.rmse()
    test.mae()
    test.rec([5, 10, 15, 20, 30, 50])
