# train_ai_model.py
import os
import django
import pandas as pd
import ta
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # 모델 저장을 위함
from datetime import timedelta, date

# Django 환경 설정
# 이 스크립트가 Django 프로젝트의 루트 디렉토리에서 실행된다고 가정합니다.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings') # 'myproject'를 실제 프로젝트 이름으로 변경하세요.
django.setup()

from stock_analyzer.models import Stock, StockDailyPrice # 모델 임포트

print("Django 환경 설정 완료. 모델 학습을 시작합니다.")

# --- 지표 계산 함수 (views.py와 동일) ---
def calculate_psar_manual(df, af_start=0.02, af_increment=0.02, af_max=0.20):
    sar_values = [np.nan] * len(df)
    if len(df) < 3:
        return pd.Series(sar_values, index=df.index)

    # 초기화 로직은 views.py의 calculate_psar_manual 함수와 동일합니다.
    # ... (생략 - views.py의 calculate_psar_manual 코드 그대로 사용) ...
    if df['Close'].iloc[1] > df['Close'].iloc[0]:
        trend = 1
        sar = df['Low'].iloc[0]
        ep = df['High'].iloc[0]
    else:
        trend = -1
        sar = df['High'].iloc[0]
        ep = df['Low'].iloc[0]
    
    af = af_start

    sar_values[0] = sar

    for i in range(1, len(df)):
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]

        prev_sar = sar_values[i-1]
        
        if trend == 1:
            sar_calc = prev_sar + af * (ep - prev_sar)
            
            if i >= 1:
                sar_calc = min(sar_calc, df['Low'].iloc[i-1])
            if i >= 2:
                sar_calc = min(sar_calc, df['Low'].iloc[i-2])
            
            sar = sar_calc

            if current_high > ep:
                ep = current_high
                af = min(af + af_increment, af_max)
            
            if current_low < sar:
                trend = -1
                sar = ep
                ep = current_low
                af = af_start
        else:
            sar_calc = prev_sar - af * (prev_sar - ep)
            
            if i >= 1:
                sar_calc = max(sar_calc, df['High'].iloc[i-1])
            if i >= 2:
                sar_calc = max(sar_calc, df['High'].iloc[i-2])
            
            sar = sar_calc
            
            if current_low < ep:
                ep = current_low
                af = min(af + af_increment, af_max)

            if current_high > sar:
                trend = 1
                sar = ep
                ep = current_high
                af = af_start

        sar_values[i] = sar

    return pd.Series(sar_values, index=df.index)

# --- 데이터 준비 및 특징(Feature) 추출 함수 ---
def prepare_data_for_ai(df, holding_period=5, profit_threshold=0.03):
    df_copy = df.copy()
    
    # 기본 기술 지표 계산
    df_copy['SAR'] = calculate_psar_manual(df_copy)
    df_copy['RSI'] = ta.momentum.rsi(df_copy['Close'], window=14, fillna=False)
    df_copy['RSI_SMA_Short'] = ta.trend.sma_indicator(df_copy['RSI'], window=5, fillna=False)
    df_copy['RSI_SMA_Long'] = ta.trend.sma_indicator(df_copy['RSI'], window=20, fillna=False)
    df_copy['STOCH_K'] = ta.momentum.stoch(df_copy['High'], df_copy['Low'], df_copy['Close'], window=14, fillna=False)
    df_copy['STOCH_D'] = ta.momentum.stoch_signal(df_copy['High'], df_copy['Low'], df_copy['Close'], window=14, smooth_window=3, fillna=False)
    
    adx_indicator = ta.trend.ADXIndicator(high=df_copy['High'], low=df_copy['Low'], close=df_copy['Close'], window=14, fillna=False)
    df_copy['DMI_plus_di'] = adx_indicator.adx_pos()
    df_copy['DMI_minus_di'] = adx_indicator.adx_neg()
    df_copy['DMI_adx'] = adx_indicator.adx()
    
    df_copy['SMA_224'] = ta.trend.sma_indicator(df_copy['Close'], window=224, fillna=False)
    df_copy['VMA'] = df_copy['Volume'].rolling(window=20, min_periods=1).mean()

    # AI 모델을 위한 특징(Features) 생성
    features = pd.DataFrame(index=df_copy.index)

    # 1. 각 지표의 현재 값
    features['RSI'] = df_copy['RSI']
    features['STOCH_K'] = df_copy['STOCH_K']
    features['STOCH_D'] = df_copy['STOCH_D']
    features['DMI_ADX'] = df_copy['DMI_adx']
    features['DMI_Plus_Minus_Diff'] = df_copy['DMI_plus_di'] - df_copy['DMI_minus_di']
    features['Close_vs_SAR_Diff'] = df_copy['Close'] - df_copy['SAR']
    features['Close_vs_SMA224_Ratio'] = df_copy['Close'] / df_copy['SMA_224']
    features['Volume_vs_VMA_Ratio'] = df_copy['Volume'] / df_copy['VMA']

    # 2. 지표의 변화량 (Momentum Features)
    features['RSI_Change_5d'] = df_copy['RSI'].diff(5)
    features['STOCH_K_Change_5d'] = df_copy['STOCH_K'].diff(5)
    features['Close_Change_5d'] = df_copy['Close'].pct_change(5) # 5일 주가 변화율

    # 3. 1단계 조건 충족 여부 (Boolean Features)
    # 현재 views.py 로직에서 사용하는 'Recently_Met' 플래그는 미래 시점을 포함하므로,
    # 여기서는 'Current' 플래그를 사용하거나, 과거 시점 기준의 'Recently_Met'를 직접 계산
    # 여기서는 학습 데이터셋을 만들기 위해 Current 지표들을 조합하여 사용
    
    # SAR 매수 추세
    features['SAR_Buy_Trend'] = (df_copy['Close'] > df_copy['SAR']).astype(int)

    # RSI 골든 크로스 발생 여부
    features['RSI_GC'] = ((df_copy['RSI_SMA_Short'] > df_copy['RSI_SMA_Long']) & \
                          (df_copy['RSI_SMA_Short'].shift(1) <= df_copy['RSI_SMA_Long'].shift(1))).astype(int)

    # RSI 매수 범위 (0~45) 내에 있는지
    features['RSI_In_Buy_Zone'] = ((df_copy['RSI'] >= 0) & (df_copy['RSI'] <= 45)).astype(int) # views.py의 rsi_buy_zone_min/max 참조

    # RSI 고점 하락 추세 (제외 조건의 반대, 즉 매수 가능한 상황)
    # views.py에서 rsi_overbought_zone_min=70을 사용하므로 동일하게 적용
    features['RSI_Not_High_Downtrend'] = ~((df_copy['RSI_SMA_Short'] < df_copy['RSI_SMA_Short'].shift(1)) & 
                                          (df_copy['RSI'] >= 70)).astype(int) # 하락 추세가 아님을 나타냄

    # Stoch 골든 크로스 발생 여부
    features['STOCH_GC'] = ((df_copy['STOCH_K'] > df_copy['STOCH_D']) & \
                            (df_copy['STOCH_K'].shift(1) <= df_copy['STOCH_D'].shift(1))).astype(int)

    # DMI 매수 조건
    features['DMI_Condition'] = ((df_copy['DMI_plus_di'] > df_copy['DMI_minus_di']) & \
                                 (df_copy['DMI_adx'] >= 15)).astype(int) # views.py의 dmi_adx_min 참조

    # 224일 이평선 근접 및 돌파 조건
    ma_224_tolerance = 0.1 # views.py의 ma_224_tolerance_percent 참조
    features['MA224_Condition'] = ((df_copy['Close'] >= df_copy['SMA_224'] * (1 - ma_224_tolerance)) & \
                                   (df_copy['Close'] <= df_copy['SMA_224'] * (1 + ma_224_tolerance)) & \
                                   ((df_copy['Close'] > df_copy['SMA_224']) | (df_copy['Low'] <= df_copy['SMA_224']))).astype(int)

    # 거래량 조건
    volume_increase_ratio = 1.2 # views.py의 volume_increase_ratio 참조
    features['Volume_Condition'] = ((df_copy['VMA'] > 0) & (df_copy['Volume'] >= df_copy['VMA'] * volume_increase_ratio)).astype(int)

    # 4. 레이블 (성공/실패) 생성
    # hold_period 일 후의 종가를 가져와 수익률 계산
    df_copy['Future_Close'] = df_copy['Close'].shift(-holding_period)
    df_copy['Profit_Loss_Percent'] = ((df_copy['Future_Close'] - df_copy['Close']) / df_copy['Close'])
    
    # profit_threshold 이상 상승했으면 1 (성공), 아니면 0 (실패)
    features['Label'] = (df_copy['Profit_Loss_Percent'] >= profit_threshold).astype(int)

    # NaN 값 제거
    features.dropna(inplace=True)
    return features.drop(columns=['Label']), features['Label']

# --- 모델 학습 및 저장 ---
def train_and_save_model(model_path='ai_stock_predictor_xgboost_model.joblib'):
    all_stocks = Stock.objects.all()
    
    # 모든 종목의 데이터를 취합 (방대한 데이터일 수 있으므로 주의)
    # 실제 운용에서는 특정 종목군만 사용하거나, 데이터량을 조절해야 합니다.
    data_frames = []
    
    # 학습 데이터 기간 설정 (예: 최근 5년간)
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * 5) # 5년치 데이터

    # 실제 사용 시, 학습 데이터의 양과 종목 수를 신중하게 선택하세요.
    # 모든 종목을 다 학습하는 것은 시간이 오래 걸리고 메모리 문제가 발생할 수 있습니다.
    # 여기서는 예시를 위해 500개 종목으로 제한
    for i, stock in enumerate(all_stocks[:500]): # 상위 500개 종목만 사용 (예시)
        if i % 100 == 0:
            print(f"{i}/{len(all_stocks[:500])} 종목 데이터 처리 중: {stock.symbol}")
        
        daily_prices = StockDailyPrice.objects.filter(stock=stock, date__range=(start_date, end_date)).order_by('date')
        if not daily_prices.exists():
            continue

        df_stock = pd.DataFrame(list(daily_prices.values(
            'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume'
        )))
        df_stock.rename(columns={
            'open_price': 'Open', 'high_price': 'High', 'low_price': 'Low',
            'close_price': 'Close', 'volume': 'Volume'
        }, inplace=True)
        df_stock.set_index('date', inplace=True)
        df_stock.sort_index(ascending=True, inplace=True)

        if len(df_stock) < 224 + 5 + 1: # 최소 데이터 길이 (가장 긴 SMA 224일 + 미래 5일 예측 + 지표 계산을 위한 여유)
            continue
        
        try:
            X, y = prepare_data_for_ai(df_stock)
            if not X.empty:
                data_frames.append(X)
                data_frames.append(y.to_frame(name='Label')) # 레이블도 DataFrame으로 추가
        except Exception as e:
            print(f"Error preparing data for {stock.symbol}: {e}")
            continue

    if not data_frames:
        print("학습할 데이터가 충분하지 않습니다. 스크립트를 종료합니다.")
        return

    # 모든 데이터를 하나의 DataFrame으로 합치기
    # X와 y가 각각의 DataFrame 리스트로 저장되었다고 가정
    X_all = pd.concat([df for df in data_frames if 'Label' not in df.columns])
    y_all = pd.concat([df['Label'] for df in data_frames if 'Label' in df.columns])

    if X_all.empty or y_all.empty:
        print("합쳐진 데이터가 비어 있습니다. 학습을 진행할 수 없습니다.")
        return
    
    # 학습 데이터와 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

    print(f"\n총 학습 데이터 샘플 수: {len(X_train)}")
    print(f"총 테스트 데이터 샘플 수: {len(X_test)}")
    print(f"학습 데이터의 긍정 샘플 비율 (Label=1): {y_train.sum() / len(y_train):.2f}")
    
    # XGBoost 분류기 모델 설정 및 학습
    # `scale_pos_weight`는 불균형한 클래스 (성공 vs 실패)를 처리하는 데 도움을 줍니다.
    # 실패(0)가 성공(1)보다 훨씬 많을 경우 긍정 클래스에 가중치를 부여합니다.
    # (총 샘플 수 - 긍정 샘플 수) / 긍정 샘플 수
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1

    model = xgb.XGBClassifier(
        objective='binary:logistic', # 이진 분류
        eval_metric='logloss',       # 평가 지표
        use_label_encoder=False,     # 경고 메시지 방지 (새로운 버전 기본값)
        n_estimators=100,            # 트리의 개수
        learning_rate=0.1,           # 학습률
        max_depth=5,                 # 트리의 최대 깊이
        subsample=0.8,               # 각 트리 구축에 사용할 샘플 비율
        colsample_bytree=0.8,        # 각 트리 구축에 사용할 피처(컬럼) 비율
        gamma=0.1,                   # 리프 노드의 추가 분할을 위한 최소 손실 감소
        random_state=42,
        scale_pos_weight=scale_pos_weight # 클래스 불균형 처리
    )

    print("\nXGBoost 모델 학습 중...")
    model.fit(X_train, y_train)
    print("XGBoost 모델 학습 완료.")

    # 모델 평가
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # 성공 확률

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\n모델 정확도: {accuracy:.4f}")
    print("\n분류 보고서:\n", report)
    print("\n혼동 행렬:\n", conf_matrix)

    # 모델 저장
    joblib.dump(model, model_path)
    print(f"\n모델이 '{model_path}'에 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    # 'myproject'를 실제 Django 프로젝트 이름으로 변경하세요!
    # 예: your_project_name.settings
    # os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project_name.settings') 
    
    # 모델 저장 경로 설정 (현재 스크립트와 동일한 디렉토리)
    MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(MODEL_DIR, 'ai_stock_predictor_xgboost_model.joblib')

    train_and_save_model(model_path=MODEL_PATH)