#%%0. 전처리된 데이터 불러오기
import pandas as pd

data = pd.read_excel('preprocessing_data.xlsx.xlsx')
data.head()

#%%1. TF-IDF로 데이터 벡터화
from sklearn.feature_extraction.text import TfidfVectorizer

# NaN 값을 빈 문자열로 대체
data['tokenized_text'] = data['tokenized_text'].fillna('')

# TF-IDF 벡터화 with Parameter Tuning
tfidf_vectorizer = TfidfVectorizer(
    min_df=0.0,      # 최소 문서 빈도 설정을 0으로 변경
    max_df=1.0,      # 최대 문서 빈도 설정을 1로 변경
    max_features=50, # 최대 피처 수
    ngram_range=(1, 2)  # n-gram 범위 설정
)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['tokenized_text'])

# 결과 확인
print("Corrected DataFrame:")
print(data)
print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())

#%%2. 클러스터링 Using DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# TF-IDF Matrix에서 cosine similarity 계산
cosine_sim = cosine_similarity(tfidf_matrix)

# 튜닝할 파라미터 값들
eps_values = [0.1, 0.5, 1.0]
min_samples_values = [2, 3, 5]

best_cluster_labels = None
best_eps = None
best_min_samples = None
best_num_clusters = 0

# 모든 조합에 대해 실험
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        dbscan_labels = dbscan.fit_predict(cosine_sim)

        # 클러스터의 수 계산
        num_clusters = len(np.unique(dbscan_labels)) - 1  # -1은 noise를 제외하기 위함

        # 클러스터의 수가 이전보다 많으면 업데이트
        if num_clusters > best_num_clusters:
            best_num_clusters = num_clusters
            best_cluster_labels = dbscan_labels
            best_eps = eps
            best_min_samples = min_samples

# 최적의 클러스터 결과를 데이터프레임에 추가
data['cluster'] = best_cluster_labels

# 최적의 파라미터와 클러스터 결과 출력
print(f"Best Parameters: eps={best_eps}, min_samples={best_min_samples}")
print("Clustered DataFrame:")
print(data)

#%%3. 차원축소 후 시각화
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# t-SNE를 사용하여 2차원으로 차원 축소
tsne = TSNE(n_components=2, random_state=42)
embedded_data = tsne.fit_transform(cosine_sim)

# 시각화
plt.figure(figsize=(8, 6))
scatter = plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=df['cluster'], cmap='viridis', alpha=0.7)
plt.title('t-SNE Visualization of Clusters')
plt.legend(*scatter.legend_elements(), title='Clusters')
plt.show()