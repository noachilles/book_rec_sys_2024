from test_util.models import RecommendResult, Dataset
from test_src.base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np
import implicit
from scipy.sparse import lil_matrix

np.random.seed(0)


class IMFRecommender(BaseRecommender):
    
    def debug_model_info(self, model, matrix):
        print("Item factors shape:", model.item_factors.shape)
        print("User factors shape:", model.user_factors.shape)
        print("Input matrix shape:", matrix.shape)
    
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 인자 수
        factors = kwargs.get("factors", 10)
        # 평가 수 임계값
        minimum_num_rating = kwargs.get("minimum_num_rating", 0)
        # 에폭 수
        n_epochs = kwargs.get("n_epochs", 50)
        # alpha: 신뢰도 연산에 필요함
        alpha = kwargs.get("alpha", 1.0)

        # MF와 마찬가지로 평가 수 임계값 이상인 영화에 대한 데이터 생성
        filtered_movielens_train = dataset.train.groupby("movie_id").filter(
            lambda x: len(x["movie_id"]) >= minimum_num_rating
        )

        # \bar{r_{ui}} 만들기 위함 - 여기서는 r_{ui} >= 4 / r_{ui} < 4 로 구분
        movielens_train_high_rating = filtered_movielens_train[dataset.train.rating >= 4]

        unique_user_ids = sorted(movielens_train_high_rating.user_id.unique())
        unique_movie_ids = sorted(movielens_train_high_rating.movie_id.unique())
        user_id2index = dict(zip(unique_user_ids, range(len(unique_user_ids))))
        movie_id2index = dict(zip(unique_movie_ids, range(len(unique_movie_ids))))

        # 희소행렬 만드는 건가...?
        movielens_matrix = lil_matrix((len(unique_movie_ids), len(unique_user_ids)))
        for i, row in movielens_train_high_rating.iterrows():
            user_index = user_id2index[row["user_id"]]
            movie_index = movie_id2index[row["movie_id"]]
            movielens_matrix[movie_index, user_index] = 1.0 * alpha

        # 모델 초기화 - ALS: 병렬 계산 지원(p, q 번갈아가며 최적값 찾는 방식)
        model = implicit.als.AlternatingLeastSquares(
            factors=factors, iterations=n_epochs, calculate_training_loss=True, random_state=1
        )

        # 학습
        model.fit(movielens_matrix)
        
        self.debug_model_info(model, movielens_matrix)

        # 추천
        recommendations = model.recommend_all(movielens_matrix.T, filter_already_liked_items=False)
        pred_user2items = defaultdict(list)
        for user_id, user_index in user_id2index.items():
            movie_indexes = recommendations[user_index, :]
            for movie_index in movie_indexes:
                movie_id = unique_movie_ids[movie_index]
                pred_user2items[user_id].append(movie_id)
        # IMF에서는 평가값의 예측이 어려우므로 rmse 평가는 수행하지 않는다(편의상, 테스트 데이터를 그대로 반환한다).
        return RecommendResult(dataset.test.rating, pred_user2items)


if __name__ == "__main__":
    IMFRecommender().run_sample()
