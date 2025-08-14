############################################################
## CHAPTER1: MEASUREMENT OF CONSTRUCTS
############################################################

### CHAPTER1(0)
# tidyverse 패키지 불러오기: 데이터 처리 및 시각화를 위해 필요
library(tidyverse)

### CHAPTER1(1)
# 제7차 근로환경조사(2023) 데이터 일부를 불러오기
# data.csv 파일을 데이터프레임으로 저장
setwd("C:/Users/wogns/OneDrive/바탕 화면") # 경로 설정 (사용자 환경에 맞게 수정 필요)
data <- read.csv("data.csv")
head(data) # 데이터의 처음 몇 줄 확인

### CHAPTER1(2)
# 계약 기간 정보가 없거나, 1개월 미만(코드 777)인 경우 제거
filtered_data <- data[!is.na(data$emp_con_period_r) & data$emp_con_period_r != 777, ]
head(filtered_data)

### CHAPTER1(3)
# 1인 근로자 분석 대상 제외
filtered_data <- filtered_data %>%
  filter(!(comp_size2 == 1 | comp_size4 == 1) | is.na(comp_size2) | is.na(comp_size4))
head(filtered_data)

### CHAPTER1(4)
# 임금 근로자만 남기고 임시 근로자 제거
filtered_data <- filtered_data %>%
  filter(emp_type == 3 & emp_stat != 2)
head(filtered_data)

### CHAPTER1(5)
# 직장 상사가 없는 경우 제외
filtered_data <- filtered_data %>%
  filter(!is.na(emp_boss_gender))
head(filtered_data)

### CHAPTER1(6)
# 분석 대상 변수 선택: 심리적 웰빙, 직무 만족, 위험노출 관련 변수
variables <- c(
  "hazard_phy1", "hazard_phy2", "hazard_phy5", "hazard_phy6", "hazard_phy7", "hazard_phy8", "hazard_phy9",
  "wsituation1", "wsituation2", "wsituation7", "wsituation8", "wsituation9", "wsituation10", "wsituation11",
  "who1", "who2", "who3", "who4", "who5"
)
selected_data <- filtered_data %>%
  select(all_of(variables))
head(selected_data)

### CHAPTER1(7)
# 결측값 처리
# 심리적 웰빙 문항(who1~who5): 8,9 → NA
# 위험요소 문항(hazard_phy*): 8,9 → NA
# 직무 만족 문항(wsituation*): 7,8,9 → NA
psych_wellbeing_columns <- c("who1", "who2", "who3", "who4", "who5")
hazard_exposure_columns <- c("hazard_phy1", "hazard_phy2", "hazard_phy5", "hazard_phy6", "hazard_phy7", "hazard_phy8", "hazard_phy9")
job_satisfaction_columns <- c("wsituation1", "wsituation2", "wsituation7", "wsituation8", "wsituation9", "wsituation10", "wsituation11")

data_cleaned <- selected_data %>%
  mutate(across(all_of(psych_wellbeing_columns), ~ ifelse(. %in% c(8, 9), NA, .))) %>%
  mutate(across(all_of(hazard_exposure_columns), ~ ifelse(. %in% c(8, 9), NA, .))) %>%
  mutate(across(all_of(job_satisfaction_columns), ~ ifelse(. %in% c(7, 8, 9), NA, .))) %>%
  filter(complete.cases(.)) # NA를 포함한 행 제거

# 제거된 행 개수 확인
removed_rows <- nrow(selected_data) - nrow(data_cleaned)
cat("제거된 행의 개수:", removed_rows, "\n")
head(data_cleaned)

### CHAPTER1(8)
# 각 변수별 Likert 척도 분포 히스토그램 시각화
data_long <- data_cleaned %>%
  select(all_of(variables)) %>%
  gather(key = "question", value = "response")

ggplot(data_long, aes(x = response)) + 
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black", alpha = 0.7) + 
  facet_wrap(~ question, scales = "free_y", ncol = 5) + 
  labs(title = "Likert Scale Distribution", x = "Response", y = "Frequency") +
  theme_minimal() + 
  theme(strip.text = element_text(size = 10), axis.text.x = element_text(angle = 45, hjust = 1))


############################################################
## CHAPTER2: EXPLORATORY FACTOR ANALYSIS (EFA)
############################################################

### CHAPTER2(0)
# psych 패키지 불러오기: EFA 수행을 위해 필요
library(psych)

### CHAPTER2(2)
# 피어슨 상관계수 행렬 계산
correlation_matrix <- cor(data_cleaned, use = "complete.obs", method = "pearson")
correlation_matrix

### CHAPTER2(3)
# 요인 분석 적합성 검정: Bartlett, KMO
library(MASS) # cortest.bartlett 함수 포함

# Bartlett의 구형성 검정: p-value < 0.05 → 요인 분석 적합
bartlett_test <- cortest.bartlett(correlation_matrix)
print(bartlett_test)

# KMO 검정: 0.6 이상 → 요인 분석 적합
kmo_test <- KMO(correlation_matrix)
print(kmo_test)

### CHAPTER2(5)
# Parallel Analysis: 최적 요인 수 확인
fa.parallel(data_cleaned,
            fa = "fa",
            n.iter = 100,
            show.legend = TRUE,
            main = "Parallel Analysis Scree Plot")

### CHAPTER2(6)
# 탐색적 요인분석(EFA) 수행: 3요인, Oblimin 회전
fa_result <- fa(data_cleaned,
                nfactors = 3,
                rotate = "oblimin", # 요인간 상관 허용
                fm = "minres") # Minimum Residual 추정
print(fa_result)

### CHAPTER2(7)
# 요인 적재값 출력
fa_result$loadings

# 요인 해석 및 변수 의미
# MR1: 직무 만족 (Job Satisfaction)
#   - wsituation1, wsituation2, wsituation7~11 포함
#   - 직무 만족 관련 변수들이 높은 적재값을 가짐 (0.693~0.790)
# MR2: 위험요소로의 노출 (Hazard Exposure)
#   - hazard_phy1, hazard_phy2, hazard_phy5~9 포함
#   - 신체적 위험 노출 변수들이 높은 적재값을 가짐 (0.703~0.892)
# MR3: 심리적 웰빙 (Psychological Wellbeing)
#   - who1~who5 포함
#   - 심리적 웰빙 관련 변수들이 높은 적재값을 가짐 (0.857~0.920)
# SS loadings: 요인별 총 변량 설명 정도
# Proportion Var: 전체 변량에서 요인이 차지하는 비율
# Cumulative Var: 누적 변량 설명 비율

### CHAPTER2(8)
# Communalities: 요인 분석으로 설명되는 정도
# Uniquenesses: 요인 분석으로 설명되지 않는 고유 정보
fa_result$communalities
fa_result$uniquenesses

### CHAPTER2(9)
# 요인간 상관계수
# MR2-MR1: -0.0892 → 위험요소가 높을수록 직무 만족 낮음
# MR2-MR3: -0.1035 → 위험노출이 높을수록 심리적 웰빙 낮음
# MR1-MR3: 0.3305 → 직무 만족 높을수록 심리적 웰빙 높음
fa_result$Phi

### CHAPTER2(10)
# 응답자별 Factor Score 계산: 후속 연구 활용 가능
factor_scores <- fa_result$scores
head(factor_scores)
# 예: K-means 클러스터링, 예측 모델링 등 활용 가능


############################################################
## CHAPTER3: CONFIRMATORY FACTOR ANALYSIS (CFA)
############################################################

### CHAPTER3(0)
# lavaan 패키지 불러오기: CFA 수행
library(lavaan)

### CHAPTER3(1)
# CFA 모델 정의 및 적합
formula <- '
  JobSat =~ wsituation1 + wsituation2 + wsituation7 + wsituation8 +
             wsituation9 + wsituation10 + wsituation11

  Hazard =~ hazard_phy1 + hazard_phy2 + hazard_phy5 +
             hazard_phy6 + hazard_phy7 + hazard_phy8 + hazard_phy9

  Wellbeing =~ who1 + who2 + who3 + who4 + who5
'
cfa_fit <- cfa(formula, data = data, estimator = "MLR")
summary(cfa_fit, fit.measures = TRUE)

### CHAPTER3(2)
# CFA 모델 잔차 확인: 상대적으로 큰 잔차 변수 쌍 확인
residuals(cfa_fit)$cov
# wsituation8-9: 0.495 → 모델 설명 부족
# wsituation10-9: 0.503 → 모델 설명 부족
# hazard_phy2-1: 0.476 → 모델 설명 부족
