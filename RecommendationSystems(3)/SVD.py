import pandas as pd

#Загрузим данные с выборкой
df_rates=pd.read_csv('D:\Dataset\SVD\user_ratedmovies.dat',sep='\t')
df_movies=pd.read_csv("D:\Dataset\SVD\movies.dat",sep='\t',encoding='iso-8859-1')

#Просмотрим содержимое данных
print(df_rates.head())
print(df_movies.head())

from sklearn.preprocessing import LabelEncoder
#Просмотрим сколько пользователей, с какими ID есть в выборке
print(df_rates.userID.min(),df_rates.userID.max())
print(len(df_rates.userID.unique()))
print(df_rates.userID.shape)

enc_user=LabelEncoder()
enc_mov=LabelEncoder()

enc_user=enc_user.fit(df_rates.userID.values)
enc_mov=enc_mov.fit(df_rates.movieID.values)


#Отберём в таблице с фильмами лишь те, за которые голосовали пользователи
idx=df_movies.loc[:,'id'].isin(df_rates.movieID)
df_movies=df_movies.loc[idx]

#Применили LabelEncoder для удобства
df_rates.loc[:,'userID'] = enc_user.transform(df_rates.loc[:,'userID'].values)
df_rates.loc[:,'movieID'] = enc_mov.transform(df_rates.loc[:,'movieID'].values)
df_movies.loc[:, 'id'] = enc_mov.transform(df_movies.loc[:,'id'].values)
print(df_rates.head())

#Матрица схожести
from scipy.sparse import coo_matrix
R = coo_matrix((df_rates.rating.values, (df_rates.userID.values,df_rates.movieID.values)))
print(R.toarray())
print(R.toarray().shape)


#SVD
from scipy.sparse.linalg import svds
u,s,vt = svds(R,k=6)
print(u.shape)
print(s.shape)
print(vt.shape)
print(vt.T)
#Оценим методом ближайших соседей
from sklearn.neighbors import NearestNeighbors

nn=NearestNeighbors(n_neighbors=10)

v=vt.T

nn.fit(v)

_,ind = nn.kneighbors(v, n_neighbors=10)
#Просмотрим матрицу с ближайшими по схожести фильмами
print(ind[:10])

movie_titles=df_movies.sort_values('id').loc[:,'title'].values
cols=['movie']+['nn_{}'.format(i) for i in range(1,10)]
print(cols)
df_ind_nn=pd.DataFrame(data=movie_titles[ind],columns=cols)
print(df_ind_nn.head())

#Выведем ближайшие фильмы к Терминатору
idx=df_ind_nn.movie.str.contains('Terminator')

print(df_ind_nn.loc[idx].head())
