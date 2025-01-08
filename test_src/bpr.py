from test_util.models import RecommendResult, Dataset
from test_src.base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np
import implicit
from scipy.sparse import lil_matrix

np.random.seed(0)

# 전체적으로 IMF와 코드가 매우 유사 ALS, BPR 알고리즘 불러오는 부분에서만 차이를 보임
class BPRRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 인자 수
        factors = kwargs.get("factors", 10)
        # 평가값의 임계값
        minimum_num_rating = kwargs.get("minimum_num_rating", 0)
        # 에폭 스
        n_epochs = kwargs.get("n_epochs", 50)
        
        # 행렬 분해용 행렬을 작성한다.
        filtered_movielens_train = dataset.train.groupby("movie_id").filter(
            lambda x: len(x["movie_id"]) >= minimum_num_rating
        )
        
        movielens_train_high_rating = filtered_movielens_train[dataset.train.rating >= 4]
        
        unique_user_ids = sorted(movielens_train_high_rating.user_id.unique())
        unique_movie_ids = sorted(movielens_train_high_rating.movie_id.unique())
        user_id2index = dict(zip(unique_user_ids, range(len(unique_user_ids))))
        movie_id2index = dict(zip(unique_movie_ids, range(len(unique_movie_ids))))
        
        # 행렬 만들기
        movielens_matrix = lil_matrix((len(unique_movie_ids), len(unique_user_ids)))
        # high rating된 인덱스에 1 입력(IMF와 동일)
        for i, row in movielens_train_high_rating.iterrows():
            user_index = user_id2index[row["user_id"]]
            movie_index = movie_id2index[row["movie_id"]]
            movielens_matrix[movie_index, user_index] = 1.0
            
        # initialize a model
        model = implicit.bpr.BayesianPersonalizedRanking(factors=factors, iterations=n_epochs)
        
        # 학습
        model.fit(movielens_matrix)
        
        # 추천
        recommendations = model.recommend_all(movielens_matrix.T, filter_already_liked_items=False)
        pred_user2items = defaultdict(list)
        for user_id, user_index in user_id2index.items():
            movie_indexes = recommendations[user_index, :]
            for movie_index in movie_indexes:
                movie_id = unique_movie_ids[movie_index]
                pred_user2items[user_id].append(movie_id)
        # BPR에서는 평가값을 예측하기 어려우므로 RMSE 평가는 수행하지 않음(편의상, 테스트 데이터의 예측값을 그대로 반환)
        return RecommendResult(dataset.test.rating, pred_user2items)
    
if __name__ == "__main__":
    BPRRecommender().run_sample()