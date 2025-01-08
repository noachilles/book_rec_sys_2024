from test_util.models import RecommendResult, Dataset
from test_src.base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np
from surprise import SVD, Reader
import pandas as pd
from surprise import Dataset as SurpriseDataset

np.random.seed(0)

class MFRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 잠재인자 수
        factors = kwargs.get("factors", 5)
        # 평갓값의 임곗값 - 평가수가 일정 이상이 아니면 한 명이 5점 준 영화도 추천
        minimum_num_rating = kwargs.get("minimum_num_rating", 100)
        # bias
        use_bias = kwargs.get("use_bias", False)
        # learning rate - SGD때문에 필요한 건가? - 아무튼 최적화를 구해야 하므로
        lr_all = kwargs.get("lr_all", 0.005)
        # 에폭 수
        n_epochs = kwargs.get("n_epochs", 50)
        
        # 평가값 개수 > minimum_num_rating로 필터링
        filtered_movielens_train = dataset.train.groupby("movie_id").filter(
            lambda x: len(x["movie_id"]) >= minimum_num_rating                                                
        )
        
        # Surprise용도 데이터 가공
        reader = Reader(rating_scale=(0.5, 5))
        data_train = SurpriseDataset.load_from_df(
            filtered_movielens_train[["user_id", "movie_id", "rating"]], reader
        ).build_full_trainset()
        
        # Surprise로 MF 학습(여기서 SVD != 특이점 분해)
        matrix_factorization = SVD(n_factors=factors, n_epochs=n_epochs, lr_all=lr_all, biased=use_bias)
        matrix_factorization.fit(data_train)
        
        def get_top_n(predictions, n=10):
            # user-id(uid)별로 예측된 아이템 저장
            top_n = defaultdict(list)
            for uid, iid, true_r, est, _ in predictions:
                top_n[uid].append((iid, est))
                
            # 사용자별 아이템 예측 평가값순으로 나열, 상위 n개 저장
            # x[1]열에 저장된 값들이 평가값 ~ n개
            for uid, user_ratings in top_n.items():
                user_ratings.sort(key=lambda x: x[1], reverse=True)
                top_n[uid] = [d[0] for d in user_ratings[:n]]
                
            return top_n
        
        # 학습 데이터에 나오지 않은 사용자와 아이템 조합 준비
        data_test = data_train.build_anti_testset(None)
        predictions = matrix_factorization.test(data_test)
        pred_user2items = get_top_n(predictions, n=10)
        
        test_data = pd.DataFrame.from_dict(
            [{"user_id": p.uid, "movie_id": p.iid, "rating_pred": p.est} for p in predictions]
        )
        movie_rating_predict = dataset.test.merge(test_data, on=["user_id", "movie_id"], how="left")
        
        # 예측할 수 없는 위치에는 평균값 저장
        movie_rating_predict.fillna({'rating_pred':filtered_movielens_train.rating.mean()}, inplace=True)
        
        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)