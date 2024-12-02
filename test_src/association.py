# 이 코드는, Apriori를 사용해 구한 lift 값을 기준으로 사용자의 최신 별점을 반영해 영화를 추천하는 기능을 구현하고 있음

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from test_util.models import RecommendResult, Dataset
from test_src.base_recommender import BaseRecommender
from collections import defaultdict, Counter
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

np.random.seed(0)


class AssociationRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 평갓값의 임곗값
        min_support = kwargs.get("min_support", 0.1)
        min_threshold = kwargs.get("min_threshold", 1)

        # 사용자 x 영화 행렬 형식으로 변경
        user_movie_matrix = dataset.train.pivot(index="user_id", columns="movie_id", values="rating")

        # 라이브러리 사용을 위해 4 이상의 평갓값은 1, 4 미만의 평갓값은 0으로 한다
        user_movie_matrix[user_movie_matrix < 4] = 0
        user_movie_matrix[user_movie_matrix.isnull()] = 0
        user_movie_matrix[user_movie_matrix >= 4] = 1

        # 지지도가 높은 영화
        freq_movies = apriori(user_movie_matrix, min_support=min_support, use_colnames=True)
        # 어소시에이션 규칙 계산(리프트값이 높은 순으로 표시)
        rules = association_rules(freq_movies, metric="lift", min_threshold=min_threshold, num_itemsets=2)

        # 사용자가 평가한 영화를 dict 형태로 저장
        
        pred_user2items = defaultdict(list)
        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()

        # 학습용 데이터에서 평갓값이 4 이상인 것만 얻는다
        # 일단 하긴 하는데, 평갓값 4 이상인 것만 얻으면 3점 영화도 추천해주지 않는 거야? 실제로 쓰이는 알고리즘은 더 복잡하겠구나
        
        movielens_train_high_rating = dataset.train[dataset.train.rating >= 4]

        for user_id, data in movielens_train_high_rating.groupby("user_id"):
            # 사용자가 직전에 평가한 5개의 영화를 얻는다.-5~인 건 abscent 정렬을 사용하기 때문인듯
            input_data = data.sort_values("timestamp")["movie_id"].tolist()[-5:]
            # 그 영화들이 조건부에 하나라도 포함되는 어소시에이션 규칙을 검출
            # x row에 대해 input_data이 비어있지 않고, x row 역시 값을 가지고 있는지 확인해 0 or 1로 matched_flags에 대입
            # 즉, matched_flags가 1인 영화를 기준으로 또 연산을 할 것으로 보임
            matched_flags = rules.antecedents.apply(lambda x: len(set(input_data) & x)) >= 1

            # association rule의 귀결부 영화를 리스트에 저장
            consequent_movies = []
            for i, row in rules[matched_flags].sort_values("lift", ascending=False).iterrows():
                consequent_movies.extend(row["consequents"])
            # 등록 빈도 세기
            counter = Counter(consequent_movies)
            for movie_id, movie_cnt in counter.most_common():
                if movie_id not in user_evaluated_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                # 추천 리스트가 10이 되면 종료한다
                if len(pred_user2items[user_id]) == 10:
                    break

        # 어소시에이션 규칙에서는 평갓값을 예측하지 않으므로, rmse 평가는 수행하지 않는다(편의상, 테스트 데이터의 예측값을 그대로 반환).
        return RecommendResult(dataset.test.rating, pred_user2items)


if __name__ == "__main__":
    AssociationRecommender().run_sample()
