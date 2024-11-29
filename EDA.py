import pandas as pd
# 영화 정보 로딩(10681)
# movieID와 제목만 사용
m_cols = ['movie_id', 'title', 'genre']
# dat 파일 형식이 어떤 건지 모르겠네(csv 같은 느낌인가?)
movies = pd.read_csv('data/ml-10m/movies.dat', names=m_cols, sep='::', encoding='latin-1', engine='python')  
# read movies.dat and adjust the cols name as m_cols. 

# genre를 list 형식으로 저장한다.
# 와 진심 처음 보는 형태의 코드...
movies['genre'] = movies.genre.apply(lambda x:x.split('|'))
print(movies.head())