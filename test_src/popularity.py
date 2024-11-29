from test_util.models import RecommendResult, Dataset
from test_src.base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np

np.random.seed(0)


class PopularityRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 평갓값의 임곗값 - 실행 파일에서 설정
        minimum_num_rating = kwargs.get("minimum_num_rating", 200)

        # movie_id에 따라 rating의 mean 값을 구한다(영화(아이템)별 평균 평갓값을 구한다.)
        movie_rating_average = dataset.train.groupby("movie_id").agg({"rating": np.mean})
        # 테스트 데이터 예측값을 저장한다. 
        # movie_id가 같을 경우에 테스트 데이터에 위의 평균값을 함께 기록한다.
        # 테스트 데이터에만 존재하는 아이템의 예측 평갓값은 0으로 한다(fillna(0)) - merge: SQL-JOIN
        movie_rating_predict = dataset.test.merge(
            movie_rating_average, on="movie_id", how="left", suffixes=("_test", "_pred")
        ).fillna(0)

        # 각 사용자에 대한 추천 영화는 해당 사용자가 아직 평가하지 않은 영화 중에서 평균값이 높은 10개 작품으로 한다
        # - movie_rating_average, 
        # 단, 평가 건수가 적으면 노이즈가 커지므로 minimum_num_rating건 이상 평가가 있는 영화로 한정한다
        # - minimum_num_rating 사용
        pred_user2items = defaultdict(list)
        # 사용자가 평가가 있는 영화에 대해 to_dict() 형태로 저장
        user_watched_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        # 아이템별 평가 평균값과 평가수(배열 크기)를 저장함
        movie_stats = dataset.train.groupby("movie_id").agg({"rating": [np.size, np.mean]})
        # minimum_num_rating 이상의 평가수를 가진 movie_stats
        # movie_stats ~: 이 부분이 flag로 작용함
        atleast_flg = movie_stats["rating"]["size"] >= minimum_num_rating
        # atleast flg 기준을 만족하는 영화에 대해 rating, mean 기준으로 내림차순 정렬하고 이를 index로 만듦
        movies_sorted_by_rating = (
            movie_stats[atleast_flg].sort_values(by=("rating", "mean"), ascending=False).index.tolist()
        )

        # 학습 데이터의 유저에 대해
        for user_id in dataset.train.user_id.unique():
            # 평균 별점이 높은 영화 중
            for movie_id in movies_sorted_by_rating:
                # 사용자가 시청한 적 없는 영화를
                if movie_id not in user_watched_movies[user_id]:
                    # 예측 리스트에 추가함
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break

        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)


if __name__ == "__main__":
    PopularityRecommender().run_sample()
