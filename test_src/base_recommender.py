# 각 알고리즘은 BaseRecommender 클래스를 상속하는 형태로 구현한다.

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from abc import ABC, abstractmethod
from test_util.data_loader import DataLoader
from test_util.metric_calculator import MetricCalculator
from test_util.models import Dataset, RecommendResult


from typing import Dict, List
class BaseRecommender(ABC):
    @abstractmethod
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        pass
    
    # 소량의 학습 데이터로 알고리즘 실행해 결과 확인할 수 있음
    def run_sample(self) -> None:
        # MovieLens의 데이터 취득
        movielens = DataLoader(num_users=1000, num_test_items=5, data_path="data/ml-10m/").load()
        
        # 추천 계산
        recommend_result = self.recommend(movielens)
        # 추천 결과 평가
        metrics = MetricCalculator().calc(
            movielens.test.rating.tolist(),
            recommend_result.rating.tolist(),
            movielens.test_user2items,
            recommend_result.user2items,
            k=10,
        )
        print(metrics)
        
