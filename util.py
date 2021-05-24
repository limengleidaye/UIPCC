import pandas as pd
import numpy as np
from tqdm import tqdm


class DataSet:
    def __init__(self, file='./data/ml-1m/ratings.dat'):
        np.random.seed(0)
        self.data_df = pd.read_csv(file, sep="::", engine='python',
                                   names=['UserId', 'MovieId', 'Rating', 'Timestamp'])

    @staticmethod
    def sparseFeature(feat, feat_num):
        return {'feat': feat, 'feat_num': feat_num}

    @staticmethod
    def denseFeature(feat):
        return {'feat': feat}

    def create_explicit_ml_1m_dataset(self, test_size=0.2):
        # =======================参数=============================
        alpha = 0.5  # 噪声的均匀分布幅度
        # ===================归一化================================
        self.data_df['user_avg'] = self.data_df.groupby('UserId')['Rating'].transform('mean')
        self.data_df['user_std'] = self.data_df.groupby('UserId')['Rating'].transform('std')
        self.data_df['norm_R'] = (self.data_df['Rating'] - self.data_df['user_avg']) / self.data_df['user_std']
        self.data_df['norm_noise_R'] = self.data_df['norm_R'] + np.random.uniform(low=-alpha, high=alpha,
                                                                                  size=len(self.data_df))

        # 划分训练集和测试集
        watch_count = self.data_df.groupby(by='UserId')['MovieId'].agg('count')  # 用户观看电影次数
        test_df = pd.concat([
            self.data_df[self.data_df.UserId == i].iloc[int((1 - test_size) * watch_count[i]):] for i in
            tqdm(watch_count.index)],
            axis=0)
        user_num, item_num = self.data_df['UserId'].max(), self.data_df['MovieId'].max()
        feature_columns = [DataSet.sparseFeature('user_id', user_num),
                           DataSet.sparseFeature('item_id', item_num)]
        test_df = test_df.reset_index()
        train_df = self.data_df.drop(labels=test_df['index'])
        train_df = train_df.drop(['Timestamp'], axis=1).sample(frac=1.).reset_index(drop=True)
        test_df = test_df.drop(['index', 'Timestamp'], axis=1).sample(frac=1.).reset_index(drop=True)
        train = train_df[['UserId', 'MovieId', 'norm_noise_R']].values
        test = test_df[['UserId', 'MovieId', 'Rating']].values
        return feature_columns, train, test

    def get_mean_std(self):
        return self.data_df[['UserId', 'user_avg', 'user_std']].drop_duplicates()
