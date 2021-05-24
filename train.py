import pandas as pd
import numpy as np
from util import DataSet
from tqdm import tqdm
import model
import random


class Evaluate:
    def __init__(self, model):
        # ==================获取训练集 测试集========================
        self.model = model
        item_num = model.movie_count
        self.all_items = set(np.arange(1, item_num + 1))
        self.train = model.trainSet
        self.test = model.testSet
        # ====================获取用户方差，均值=====================
        self.user_avg_std = model.user_avg_std
        return

    def recommend(self, user, list):
        return [item[0] for item in
                sorted(self.model.predict(user, list).items(), key=lambda x: x[1], reverse=True)]

    def rec(self, N):
        print("Evaluation start ...", N)
        # 回率
        test = {}
        for user, movies in self.test.items():
            for movie in movies:
                if self.test[user][movie] > 4:
                    test[user] = set()
                    test[user].add(movie)
        # rec_count = 0
        # ==============================
        test_count = 0
        hits = {}
        for n in N:
            hits.setdefault(n, 0)
        # ==========================================

        for user in tqdm(test.keys()):
            train_items = set(self.train[user].keys())
            test_items = test[user]
            other_items = self.all_items - train_items.union(test_items)
            for idx in test_items:
                random_items = random.sample(other_items, 200)
                random_items.append(idx)
                rec_movies = self.recommend(user, random_items)
                for n in N:
                    hits[n] += int(idx in rec_movies[:n])
                    # rec_count += n
                test_count += len(test_items)
                # precision = hit / (1.0 * rec_count)
        for n in N:
            recall = hits[n] / (1.0 * test_count)
            print('N:%d\trecall=%.6f\t' % (n, recall))

    def rmse_and_mae(self):
        test_list = []
        y_true = []
        y_pred = []

        for user, movies in self.test.items():
            for movie in movies:
                test_list.append([user, movie])
                y_true.append(movies[movie])
        predict_y = self.model.evaluate(test_list)
        # ====================评分乘上标准差加上平均值====================
        for [user, rating] in predict_y:
            pred_r = rating * self.user_avg_std[user]['user_std'] + self.user_avg_std[user][
                'user_avg']
            y_pred.append(np.clip(pred_r, 1, 5) if pred_r < 0 or pred_r > 6 else pred_r)
        rmse = np.sqrt(np.sum(np.power(np.array(y_true) - np.array(y_pred), 2)) / len(y_true))
        mse = np.sum(np.abs(np.array(y_true) - np.array(y_pred))) / len(y_true)
        print('rmse:%.6f\tmse:%.6f' % (rmse, mse))


if __name__ == '__main__':
    # =================train model======================
    # model.IPCC()
    #==================选择模型=============================
    test = Evaluate(model.UIPCC())
    test.rmse_and_mae()
    test.rec([5, 10, 15, 20, 30, 50])
