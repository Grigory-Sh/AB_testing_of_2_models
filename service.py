import os
import pickle
import hashlib
import psycopg2
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from typing import List
from schema import PostGet
from datetime import datetime
from fastapi import Depends, FastAPI
from sqlalchemy import create_engine
from psycopg2.extras import RealDictCursor

from schema import PostGet, Response

app = FastAPI()

def get_model_path(group, path: str) -> str:
    # проверяем где выполняется код в лмс, или локально. Немного магии
    if os.environ.get("IS_LMS") == "1":
        if group == 'control':
            MODEL_PATH = '/workdir/user_input/model_control'
        elif group == 'test':
            MODEL_PATH = '/workdir/user_input/model_test'
        else:
            raise ValueError('unknown group')
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models(group):
    # LOAD MODEL HERE PLS :)
    if group == 'control':
        filename = './model_control.pkl'
    elif group == 'test':
        filename = './model_test.pkl'
    else:
        raise ValueError('unknown group')
    model_path = get_model_path(group, filename)
    loaded_model = pickle.load(open(model_path, 'rb'))
    return loaded_model

def batch_load_sql(query: str):
    engine = create_engine("postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
                           "postgres.lab.karpov.courses:6432/startml")
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=20000):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def get_exp_group(user_id: int) -> str:
    # при помощи хэшфункции и соли относим пользователя, по его id к группе control или test
    if (int(hashlib.md5((str(user_id) + "fA iu").encode()).hexdigest(), 16) % 2) == 0:
        return 'test'
    else:
        return 'control'

def control_predictions(id: int,
                        time: datetime,
                        limit: int = 10) -> List[PostGet]:

    # получаем данные пользователя по id из users
    pers = users[users['user_id']==id].drop('user_id', axis=1).iloc[0]
    # создаём DataFrame для предсказания
    pred = add_user_time(note_control, pers, time).drop('post_id', axis=1)
    # создаём DataFrame с предсказаниями
    predict = pd.DataFrame(post_text['post_id'])
    # делаем предсказания на основе pred
    predict['predict'] = model_control.predict_proba(pred)[:,1]
    # сортируем посты по предсказанной вероятности от большей к меньшей
    predict.sort_values('predict', ascending=False, inplace=True, ignore_index=True)
    # отбираем limit постов
    posts = predict.head(limit)['post_id'].values
    # формируем DataFrame с предсказанными постами
    posts = post_text[post_text['post_id'].isin(posts)]

    return [PostGet(**{"id": i[0], "text": i[1], "topic": i[2]}) for i in posts.itertuples(index=False)]

def test_predictions(id: int,
                     time: datetime,
                     limit: int = 10) -> List[PostGet]:

    # получаем данные пользователя по id из users
    pers = users[users['user_id']==id].drop('user_id', axis=1).iloc[0]
    # создаём DataFrame для предсказания
    pred = add_user_time(note_test, pers, time).drop('post_id', axis=1)
    # создаём DataFrame с предсказаниями
    predict = pd.DataFrame(post_text['post_id'])
    # делаем предсказания на основе pred
    predict['predict'] = model_test.predict_proba(pred)[:,1]
    # сортируем посты по предсказанной вероятности от большей к меньшей
    predict.sort_values('predict', ascending=False, inplace=True, ignore_index=True)
    # отбираем limit постов
    posts = predict.head(limit)['post_id'].values
    # формируем DataFrame с предсказанными постами
    posts = post_text[post_text['post_id'].isin(posts)]

    return [PostGet(**{"id": i[0], "text": i[1], "topic": i[2]}) for i in posts.itertuples(index=False)]

# cкачиваем все данные из post_text
post_text = batch_load_sql(
    "SELECT * FROM public.post_text_df"
)

# cкачиваем все подготовленные данные из note_control_shaverdin
note_control = batch_load_sql(
    "SELECT * FROM note_control_shaverdin"
)

# cкачиваем все подготовленные данные из note_test_shaverdin
note_test = batch_load_sql(
    "SELECT * FROM note_test_shaverdin"
)

# cкачиваем все подготовленные данные из users_shaverdin
users = batch_load_sql(
    "SELECT * FROM users_shaverdin"
)

# функция добавления к Dataframe данных пользователя и всремени
def add_user_time(note, pers, time):
    # добавляем колонки с данными пользователя
    note['gender'] = pers[0]
    note['age'] = pers[1]
    note['country'] = pers[2]
    note['city'] = pers[3]
    note['exp_group'] = pers[4]
    note['os'] = pers[5]
    note['source'] = pers[6]
    note['user_stat'] = pers[7]
    # добавляем колонки с данными времени
    note['year'] = time.year
    note['month'] = time.month
    note['day'] = time.day
    note['hour'] = time.hour
    note['minute'] = time.minute
    note['second'] = time.second

    return note

# загружаем модели
model_control = load_models('control')
model_test = load_models('test')

@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int,
                      time: datetime,
                      limit: int = 10) -> Response:

    # получаем группу пользователя
    group = get_exp_group(id)
    # создаём рекомендации для пользователя в зависимости от пренадлежности к группе
    if group == 'control':
        recom = control_predictions(id, time, limit)
    elif group == 'test':
        recom = test_predictions(id, time, limit)
    else:
        raise ValueError('unknown group')

    return Response(exp_group=group, recommendations=recom)
