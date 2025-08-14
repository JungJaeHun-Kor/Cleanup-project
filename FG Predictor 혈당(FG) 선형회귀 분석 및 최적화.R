# =========================================================
# FG 데이터 분석: 데이터 처리, EDA, 선형회귀, 변환 및 모형 개선
# =========================================================

# =========================================================
# 1. 패키지 로드
# =========================================================
library(tidyverse)
library(Hmisc)
library(forcats)
library(MASS)
library(car)
library(lmtest)
library(sandwich)
library(ggplot2)
library(patchwork)

# =========================================================
# 2. 데이터 불러오기
# =========================================================
setwd("C:/Users/wogns/OneDrive/바탕 화면")  # 사용자 환경에 맞게 설정
data <- read.csv("data.csv")
head(data)

# =========================================================
# 3. 데이터 구조 확인 및 수치형/범주형 확인
# =========================================================
Hmisc::describe(data)

# 문자형 변수 factor로 변환, NA 명시
data <- data %>% 
  mutate(across(where(is.character), ~ fct_explicit_na(as.factor(.), na_level = "NA")))
str(data)

# =========================================================
# 4. 초기 회귀분석 (FG ~ 모든 변수)
# =========================================================
model <- lm(FG ~ ., data = data)
summary(model)

# 다중공선성 확인
car::vif(model)

# =========================================================
# 5. 범주형 변수 통합 및 이름 축약
# =========================================================
# race 변수 통합
data <- data %>%
  mutate(race = fct_collapse(race,
                             Hispanic = c("Mexican American", "Other Hispanic"),
                             `Non-Hispanic Other` = c("Non-Hispanic Asian", "Other/Multi")))

# 범주 이름 축약
data <- data %>%
  mutate(
    income = fct_recode(income,
                        "<$25K" = "< $25,000",
                        "$25K-<$55K" = "$25,000 to < $55,000",
                        "$55K+" = "$55,000+"),
    race = fct_recode(race,
                      "H" = "Hispanic",
                      "NHW" = "Non-Hispanic White",
                      "NHB" = "Non-Hispanic Black",
                      "NHO" = "Non-Hispanic Other")
  )
head(data)

# =========================================================
# 6. EDA: 수치형 및 범주형 변수 시각화
# =========================================================
p1 <- ggplot(data, aes(x = WC, y = FG)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "Waist Circumference", y = "FG(mmd/L)")

p2 <- ggplot(data, aes(x = smoker, y = FG)) +
  geom_boxplot() +
  labs(x = "Smoking Status", y = "FG(mmd/L)")

p3 <- ggplot(data, aes(x = age, y = FG)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE)+
  labs(x = "Age (years)", y = "FG(mmd/L)") 

p4 <- ggplot(data, aes(x = gender, y = FG)) +
  geom_boxplot()+
  labs(x = "Gender", y = "FG(mmd/L)")

p5 <- ggplot(data, aes(x = race, y = FG)) +
  geom_boxplot()+
  labs(x = "Race", y = "FG(mmd/L)")

p6 <- ggplot(data, aes(x = income, y = FG)) +
  geom_boxplot() +
  labs(x = "Income", y = "FG(mmd/L)")

# 2행 3열 배치
(p1 | p2 | p3) / (p4 | p5 | p6)

# =========================================================
# 7. Box-Cox 변환: FG 변환 전 최소값 확인
# =========================================================
if(min(data$FG) <= 0) data$FG <- data$FG + abs(min(data$FG)) + 1

nmodel <- lm(FG ~ ., data = data)
summary(nmodel)

boxcox_result <- boxcox(nmodel, lambda = seq(-2, 2, 0.1))
lambda_opt <- boxcox_result$x[which.max(boxcox_result$y)]
lambda_opt_rounded <- round(lambda_opt, 1)
cat("최적의 lambda 값:", lambda_opt_rounded, "\n")

# FG 변환
data$FG_trans <- if (abs(lambda_opt) < 1e-6) {
  log(data$FG)
} else {
  (data$FG^lambda_opt - 1) / lambda_opt
}

model_trans <- lm(FG_trans ~ . -FG, data = data)
summary(model_trans)

# =========================================================
# 8. 잔차 시각화 (heteroscedasticity 확인)
# =========================================================
resid_data <- data %>% mutate(residuals = resid(model_trans))

p1 <- ggplot(resid_data, aes(x = WC, y = residuals)) +
  geom_point() +
  geom_smooth(method = "loess", se = FALSE, col = "blue") +
  labs(x = "Waist Circumference (cm)", y = "Residuals")

p2 <- ggplot(resid_data, aes(x = smoker, y = residuals)) +
  geom_boxplot() +
  labs(x = "Smoking Status", y = "Residuals")

p3 <- ggplot(resid_data, aes(x = age, y = residuals)) +
  geom_point() +
  geom_smooth(method = "loess", se = FALSE, col = "blue") +
  labs(x = "Age (years)", y = "Residuals")

p4 <- ggplot(resid_data, aes(x = gender, y = residuals)) +
  geom_boxplot() +
  labs(x = "Gender", y = "Residuals")

p5 <- ggplot(resid_data, aes(x = race, y = residuals)) +
  geom_boxplot() +
  labs(x = "Race", y = "Residuals")

p6 <- ggplot(resid_data, aes(x = income, y = residuals)) +
  geom_boxplot() +
  labs(x = "Income", y = "Residuals")

(p1 | p2 | p3) / (p4 | p5 | p6)

# =========================================================
# 9. 이상치 제거: Cook's distance 기준
# =========================================================
cookd <- cooks.distance(model_trans)
data_clean <- data[cookd < 4/(nrow(data)-length(model_trans$coefficients)-2), ]

# 불필요 변수 제거
vars_to_remove <- c("FG", "residuals")
data_clean <- data_clean[, !(names(data_clean) %in% vars_to_remove)]

# 재적합
model_clean <- lm(FG_trans ~ ., data = data_clean)
summary(model_clean)

# =========================================================
# 10. HC3 robust standard error 적용
# =========================================================
coeftest(model_clean, vcov = vcovHC(model_clean, type = "HC3"))
confint(coeftest(model_clean, vcov=vcovHC(model_clean, type="HC3")))
