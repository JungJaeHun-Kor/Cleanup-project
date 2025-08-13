# =========================================================
# 통합 Python 분석 코드 (서울 상권 + 오피스 데이터)
# =========================================================

# ==================== 라이브러리 불러오기 ====================
import pandas as pd
import numpy as np
import requests, json
import folium
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 1. 데이터 불러오기 및 전처리 (상권 데이터)
# =========================================================
df_gu = pd.read_csv('data_gu.csv')
df_gu.columns = ['STD_YM', 'DO_NM', 'GU_NM', 'GU_CD', 'CLASS', 'SALE', 'FLOW_POP']
df_gu = df_gu[['STD_YM','GU_NM','CLASS','SALE','FLOW_POP']]

# 2025년 이후 데이터 필터링
df_2025 = df_gu[df_gu['STD_YM'] >= 202501]
mean_df = df_2025.groupby('GU_NM').agg(
  mean_sale=('SALE','mean'),
  mean_flowpop=('FLOW_POP','mean')
).reset_index()

# 산점도
plt.figure(figsize=(8,6))
plt.scatter(mean_df['mean_sale'], mean_df['mean_flowpop'], color='blue')
plt.xlabel('Average Sale')
plt.ylabel('Average Flow Population')
plt.title('구별 매출과 유동인구 산점도')
plt.grid(True)
plt.show()

print(f"구별 평균 매출과 유동인구 상관계수: {mean_df['mean_sale'].corr(mean_df['mean_flowpop']):.4f}")

# 구별 업종별 총 매출
ndf_gu = df_gu.groupby(['GU_NM','CLASS']).agg(total_sale=('SALE','sum')).reset_index()

# 원형 그래프 반복
for gu in ndf_gu['GU_NM'].unique():
  temp = ndf_gu[ndf_gu['GU_NM']==gu]
plt.figure()
plt.pie(temp['total_sale'], labels=temp['CLASS'], autopct='%1.1f%%')
plt.title(gu)
plt.show()

# 의료기관 매출 pivot
pivot_table = df_gu[df_gu['CLASS']=='의료기관'].groupby(['STD_YM','GU_NM']).agg(
  sale=('SALE','sum')).reset_index()

# 특정 구 강남구 시각화
pivot_table2 = pivot_table[pivot_table['GU_NM']=='강남구']
plt.figure(figsize=(10,5))
plt.bar(pivot_table2['STD_YM'].astype(str), pivot_table2['sale'])
plt.xlabel("연월")
plt.ylabel("매출")
plt.title("강남구 의료기관 매출")
plt.xticks(rotation=45)
plt.show()

# =========================================================
# 2. Folium 지도 시각화
# =========================================================
res = requests.get("https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json")
seoul_geo = json.loads(res.content)

df_gu['FLOW_PER_POP'] = df_gu['SALE']/df_gu['FLOW_POP']
map_seoul = folium.Map(location=[37.5665,126.9780], zoom_start=11, max_bounds=True, tiles='CartoDBpositron')
folium.GeoJson(seoul_geo).add_to(map_seoul)

for cls in ['일반', '의료기관', '스포츠시설', '여행', '유흥', '자기계발']:
  df_cls = df_gu[df_gu['CLASS']==cls]
folium.Choropleth(
  geo_data=seoul_geo, data=df_cls,
  columns=['GU_NM','FLOW_PER_POP'],
  fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2,
  key_on='feature.properties.name', name=cls
).add_to(map_seoul)

# map_seoul.save("seoul_choropleth.html")  # 필요시 저장

# =========================================================
# 3. XGBoost 단변량 시계열 예측 (Chapter 2)
# =========================================================
df_gugu = df_gu[~df_gu['GU_NM'].isin(['중구','송파구','관악구','마포구','영등포구'])]
ndf_gu = df_gugu[df_gugu['CLASS']=='의료기관']
ndf_gu['STD_YM'] = pd.to_datetime(ndf_gu['STD_YM'], format='%Y%m')
df_pivot = pd.pivot_table(ndf_gu, values='SALE', index='GU_NM', columns='STD_YM', aggfunc='mean')

train = df_pivot.iloc[:,:-1]
test = df_pivot.iloc[:,-1]

xgb_model = xgb.XGBRegressor()
param_dist = {
  "n_estimators": [100,200,500],
  "learning_rate": [0.01,0.01,0.3],
  "max_depth":[3,5,7],
  "subsample":[0.7,1.0],
  "colsample_bytree":[0.7,0.9]
}
mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist,
                                   n_iter=20, scoring=mape_scorer, cv=3, verbose=1, random_state=42)
random_search.fit(train, test)

print("Best Parameters:", random_search.best_params_)
y_pred = random_search.predict(train)
print(f"MAPE: {mean_absolute_percentage_error(test,y_pred):.4f}")

# 예측 시각화
df_diff = pd.DataFrame({"GU_NM": train.index, "Actual": test.values, "Predicted": y_pred})
num_regions = len(df_diff)
fig, axes = plt.subplots(nrows=num_regions//3+1, ncols=3, figsize=(20,15))
axes = axes.flatten()
for i, (region, actual, predicted) in enumerate(zip(df_diff["GU_NM"], df_diff["Actual"], df_diff["Predicted"])):
  ax = axes[i]
ax.plot(range(len(train.columns)), train.loc[region], marker='o', color='blue', label='Actual')
ax.scatter(len(train.columns)-1, predicted, color='red', label='Pred', alpha=0.6)
ax.set_title(f"{region} 의료기관 매출 추이")
ax.legend()
plt.tight_layout()
plt.show()

# =========================================================
# 4. 오피스 데이터 분석 (Chapter 3)
# =========================================================
df_office = pd.read_csv('data_sdm.csv')
df_office = df_office[df_office['bld_type'] != '집합']

# 가격 조정
price_avg = df_office.groupby('dealt_yr')['price_tr'].mean().pct_change().fillna(0)
adjust_factor = np.cumprod(price_avg+1)

# 가격 보정
for yr, factor in zip(range(2014,2024), adjust_factor):
  df_office.loc[df_office['dealt_yr']==yr,'price_tr'] *= factor

df_office.rename(columns={'price_tr':'adjusted_price'}, inplace=True)
df_office.drop(['dealt_yr'], axis=1, inplace=True)

# 표준화
scaler = StandardScaler()
for col in ['gfa_dlt','land_area','floor_gr','floor_bm','year_con']:
  df_office[col] = scaler.fit_transform(df_office[[col]])

# condition_score 계산
df_office['condition_score'] = (df_office['gfa_dlt']*0.4015 + df_office['land_area']*0.3488 +
                                  df_office['floor_gr']*0.2237 + df_office['floor_bm']*0.0112 +
                                  df_office['year_con']*0.0013)

# 영업시간 범주화
def categorize_time(x):
  res=[]
for val in x:
  if val == '휴무일':
  res.append('closed')
else:
  h1,h2 = map(int,val.split('~')[0].split(':')), map(int,val.split('~')[1].split(':'))
diff = h2[0]-h1[0]
if diff==24:
  res.append('full time')
elif diff>13:
  res.append('most time')
else:
  res.append('half time')
return res

for day in ['MON_OPER_TIME','TUES_OPER_TIME','WED_OPER_TIME','THUR_OPER_TIME','FRI_OPER_TIME','SAT_OPER_TIME','SUN_OPER_TIME']:
  df_office[day] = categorize_time(df_office[day])

# 서비스 점수 계산
ord_dic = {"full time":3, "most time":2, "half time":1, "closed":-1}
weekday = ['MON_OPER_TIME','TUES_OPER_TIME','WED_OPER_TIME','THUR_OPER_TIME','FRI_OPER_TIME']
weekend = ['SAT_OPER_TIME','SUN_OPER_TIME']

df_office['weekday_oper_time'] = df_office[weekday].mode(axis=1)[0].map(ord_dic)
df_office['weekend_oper_time'] = df_office[weekend].mode(axis=1)[0].map(ord_dic)

# 라벨 인코딩
for col in ['STN_PROXIMITY_AT','PARKNG_POSBL_AT','WIFI_HOLD_AT','AC_HOLD_AT','HEATER_HOLD_AT']:
  le = LabelEncoder()
df_office[col+'_label'] = le.fit_transform(df_office[col])

# 최종 서비스 점수
df_office['service_score'] = (df_office['weekday_oper_time'] + df_office['weekend_oper_time'] +
                                df_office['STN_PROXIMITY_AT_label'] + df_office['PARKNG_POSBL_AT_label'] +
                                df_office['WIFI_HOLD_AT_label'] + df_office['AC_HOLD_AT_label'] +
                                df_office['HEATER_HOLD_AT_label'])

# 최종 점수 계산
df_office[['adjusted_price','condition_score','service_score']] = MinMaxScaler().fit_transform(df_office[['adjusted_price','condition_score','service_score']])
df_office['final_score'] = 0.791667*df_office['condition_score'] - 0.396371*df_office['adjusted_price'] + 0.20305*df_office['service_score']

# 최고 점수 오피스
best_office = df_office[df_office['final_score']==df_office['final_score'].max()]
best_office
