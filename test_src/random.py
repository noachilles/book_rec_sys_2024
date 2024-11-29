# 무작위 추천, 0.5~5.0 난수 발생 후 그것을 예측 평갓값으로.
# 사용자X아이템 행렬을 만들고 각 셀에 난수 저장
# pred_user2items라는 딕셔너리 작성하고 key: user_id, value: 사용자 미평가 영화 무작위 10개

from test_src.base_recommender import BaseRecommender
from test_util.models import RecommendResult, Dataset
from collections import defaultdict
import numpy as np

np.random.seed(0)

class RandomRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 사용자 ID와 아이템 ID에 대해 0부터 시작하는 인덱스 할당
        # unique() 사용하는 이유는 중복 없이 할당하기 위함 / 학습용 데이터인 dataset.train 사용
        unique_user_ids = sorted(dataset.train.user_id.unique())
        unique_movie_ids = sorted(dataset.train.movie_id.unique())
        user_id2index = dict(zip(unique_user_ids, range(len(unique_user_ids))))
        movie_id2index = dict(zip(unique_movie_ids, range(len(unique_movie_ids))))
        
        # 사용자X아이템의 행렬에서 각 셀의 예측 평갓값은 0.5~5.0의 균등 난수로 한다.  
        pred_matrix = np.random.uniform(0.5, 5.0, (len(unique_user_ids), len(unique_movie_ids)))
        
        # RMSE 평가용으로 test data에 나오는 사용자와 아이템의 예측 평갓값을 저장(RMSE 용이면 예측 별점이 맞음)  
        # 영화, 유저 정보와 함께 pred_results를 담은 movie_rating_predict(rating 값은 추후 추가)
        movie_rating_predict = dataset.test.copy()
        pred_results = []
        
        for i, row in dataset.test.iterrows():
            user_id = row["user_id"]
            
            # 학습용이 아니었던 영화에 대해서도 난수를 입력(어차피 랜덤 추천이므로 예상 별점 역시 난수로 기입함)
            if row["movie_id"] not in movie_id2index:
                pred_results.append(np.random.uniform(0.5, 5.0))
                continue
            # 학습용 데이터와 겹치는 테스트 데이터에 대해서는 
            user_index = user_id2index[row["user_id"]]
            movie_index = movie_id2index[row["movie_id"]]
            # 각 셀의 예측 평갓값을 가져와서
            pred_score = pred_matrix[user_index, movie_index]
            # 난수값을 pred_results에 입력
            pred_results.append(pred_score)
    
        movie_rating_predict["rating_pred"] = pred_results
        # IDEA::::순위 평가용 데이터 작성 -> 각 사용자에 대한 추천 영화는 해당 사용자 평가 X 영화 중 무작위 10개
        pred_user2items = defaultdict(list)
        # "user_id"별로 시청한 "movie_id"리스트를 집계해서 "movie_id"만 따로 dict로 만듦
        # == 사용자가 이미 평가한 영화 저장
        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        
        for user_id in unique_user_ids:
            user_index = user_id2index[user_id]
            # argsort?
            movie_indexes = np.argsort(-pred_matrix[user_index, :])
            for movie_index in movie_indexes:
                movie_id = unique_movie_ids[movie_index]
                # 만약 movie_id가 평가한 항목에 있지 않다면
                if movie_id not in user_evaluated_movies[user_id]:
                    # 순위 평가용 데이터에 movie_id를 추가
                    pred_user2items[user_id].append(movie_id)
                # 10개 수집
                if len(pred_user2items[user_id]) == 10:
                    break
                
        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)
    
if __name__ == "__main__":
    RandomRecommender().run_sample()
    