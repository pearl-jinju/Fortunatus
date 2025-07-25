# stock_analyzer/views.py

import pandas as pd
import ta
import numpy as np
import joblib
import os

from datetime import timedelta, date, datetime

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from .models import Stock, StockDailyPrice

# --- XGBoost 모델 로드 (서버 시작 시 한 번만 로드) ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ai_stock_predictor_xgboost_model.joblib')

ai_model = None
try:
    ai_model = joblib.load(MODEL_PATH)
    print(f"XGBoost AI 모델이 '{MODEL_PATH}'에서 성공적으로 로드되었습니다.")
except FileNotFoundError:
    print(f"경고: XGBoost AI 모델 파일 '{MODEL_PATH}'을(를) 찾을 수 없습니다. AI 보완 기능이 비활성화됩니다. 'python train_ai_model.py'를 실행하여 모델을 학습시키고 저장하십시오.")
except Exception as e:
    print(f"경고: XGBoost AI 모델 로드 중 오류 발생: {e}. AI 보완 기능이 비활성화됩니다.")
    ai_model = None

# --- PSAR을 직접 계산하는 함수 ---
def calculate_psar_manual(df, af_start=0.02, af_increment=0.02, af_max=0.20):
    sar_values = [np.nan] * len(df)

    if len(df) < 3:
        return pd.Series(sar_values, index=df.index)

    # 초기 SAR 및 EP 설정
    # 첫 캔들 이후 상승 추세로 시작하는 경우
    if df['Close'].iloc[1] > df['Close'].iloc[0]:
        trend = 1 # 1: 상승, -1: 하락
        sar = df['Low'].iloc[0]
        ep = df['High'].iloc[0]
    # 첫 캔들 이후 하락 추세로 시작하는 경우
    else:
        trend = -1
        sar = df['High'].iloc[0]
        ep = df['Low'].iloc[0]

    af = af_start # 가속 인자 (Acceleration Factor)

    sar_values[0] = sar # 첫 번째 SAR 값 초기화

    for i in range(1, len(df)):
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]

        prev_sar = sar_values[i-1] # 이전 SAR 값 가져오기

        if trend == 1: # 상승 추세
            sar_calc = prev_sar + af * (ep - prev_sar)

            # SAR이 이전 두 봉의 최저가보다 높으면 해당 최저가로 제한
            if i >= 1:
                sar_calc = min(sar_calc, df['Low'].iloc[i-1])
            if i >= 2:
                sar_calc = min(sar_calc, df['Low'].iloc[i-2])

            sar = sar_calc

            # 새로운 고점이 EP보다 높으면 EP 업데이트, AF 증가
            if current_high > ep:
                ep = current_high
                af = min(af + af_increment, af_max)

            # 현재 저가가 SAR보다 낮으면 추세 반전
            if current_low < sar:
                trend = -1
                sar = ep # SAR을 이전 EP로 설정
                ep = current_low # 새로운 EP는 현재 저가
                af = af_start # AF 초기화
        else: # 하락 추세
            sar_calc = prev_sar - af * (prev_sar - ep)

            # SAR이 이전 두 봉의 최고가보다 낮으면 해당 최고가로 제한
            if i >= 1:
                sar_calc = max(sar_calc, df['High'].iloc[i-1])
            if i >= 2:
                sar_calc = max(sar_calc, df['High'].iloc[i-2])

            sar = sar_calc

            # 새로운 저점이 EP보다 낮으면 EP 업데이트, AF 증가
            if current_low < ep:
                ep = current_low
                af = min(af + af_increment, af_max)

            # 현재 고가가 SAR보다 높으면 추세 반전
            if current_high > sar:
                trend = 1
                sar = ep # SAR을 이전 EP로 설정
                ep = current_high # 새로운 EP는 현재 고가
                af = af_start # AF 초기화

        sar_values[i] = sar # 계산된 SAR 값 저장

    return pd.Series(sar_values, index=df.index)

# --- AI 모델에 입력할 특징(Feature) 추출 함수 ---
def extract_features_for_ai_prediction(df_segment):
    df_copy = df_segment.copy()

    # AI 모델 학습에 사용된 지표와 동일하게 계산
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

    features = pd.DataFrame(index=df_copy.index)
    last_row_index = df_copy.index[-1]

    # AI 모델 학습에 사용된 특징 생성 (단일 행)
    features.loc[last_row_index, 'RSI'] = df_copy.loc[last_row_index, 'RSI']
    features.loc[last_row_index, 'STOCH_K'] = df_copy.loc[last_row_index, 'STOCH_K']
    features.loc[last_row_index, 'STOCH_D'] = df_copy.loc[last_row_index, 'STOCH_D']
    features.loc[last_row_index, 'DMI_ADX'] = df_copy.loc[last_row_index, 'DMI_adx']
    features.loc[last_row_index, 'DMI_Plus_Minus_Diff'] = df_copy.loc[last_row_index, 'DMI_plus_di'] - df_copy.loc[last_row_index, 'DMI_minus_di']
    features.loc[last_row_index, 'Close_vs_SAR_Diff'] = df_copy.loc[last_row_index, 'Close'] - df_copy.loc[last_row_index, 'SAR']
    features.loc[last_row_index, 'Close_vs_SMA224_Ratio'] = df_copy.loc[last_row_index, 'Close'] / df_copy.loc[last_row_index, 'SMA_224']
    features.loc[last_row_index, 'Volume_vs_VMA_Ratio'] = df_copy.loc[last_row_index, 'Volume'] / df_copy.loc[last_row_index, 'VMA']

    if len(df_copy) >= 5: # 5일 전 데이터가 있어야 계산 가능
        features.loc[last_row_index, 'RSI_Change_5d'] = df_copy.loc[last_row_index, 'RSI'] - df_copy.iloc[-5]['RSI']
        features.loc[last_row_index, 'STOCH_K_Change_5d'] = df_copy.loc[last_row_index, 'STOCH_K'] - df_copy.iloc[-5]['STOCH_K']
        features.loc[last_row_index, 'Close_Change_5d'] = df_copy.loc[last_row_index, 'Close'] / df_copy.iloc[-5]['Close'] - 1
    else: # 데이터가 부족하면 NaN으로 채움
        features.loc[last_row_index, ['RSI_Change_5d', 'STOCH_K_Change_5d', 'Close_Change_5d']] = np.nan

    # 이진 조건 특징 (boolean을 int로 변환)
    features.loc[last_row_index, 'SAR_Buy_Trend'] = (df_copy.loc[last_row_index, 'Close'] > df_copy.loc[last_row_index, 'SAR']).astype(int)

    features.loc[last_row_index, 'RSI_GC'] = ((df_copy.loc[last_row_index, 'RSI_SMA_Short'] > df_copy.loc[last_row_index, 'RSI_SMA_Long']) & \
                                              (df_copy.iloc[-2]['RSI_SMA_Short'] <= df_copy.iloc[-2]['RSI_SMA_Long'])).astype(int)

    features.loc[last_row_index, 'RSI_In_Buy_Zone'] = ((df_copy.loc[last_row_index, 'RSI'] >= 0) & (df_copy.loc[last_row_index, 'RSI'] <= 45)).astype(int)

    features.loc[last_row_index, 'RSI_Not_High_Downtrend'] = ~((df_copy.loc[last_row_index, 'RSI_SMA_Short'] < df_copy.iloc[-2]['RSI_SMA_Short']) &
                                                              (df_copy.loc[last_row_index, 'RSI'] >= 70)).astype(int)

    features.loc[last_row_index, 'STOCH_GC'] = ((df_copy.loc[last_row_index, 'STOCH_K'] > df_copy.loc[last_row_index, 'STOCH_D']) & \
                                                (df_copy.iloc[-2]['STOCH_K'] <= df_copy.iloc[-2]['STOCH_D'])).astype(int)

    features.loc[last_row_index, 'DMI_Condition'] = ((df_copy.loc[last_row_index, 'DMI_plus_di'] > df_copy.loc[last_row_index, 'DMI_minus_di']) & \
                                                     (df_copy.loc[last_row_index, 'DMI_adx'] >= 15)).astype(int)

    ma_224_tolerance = 0.1
    features.loc[last_row_index, 'MA224_Condition'] = ((df_copy.loc[last_row_index, 'Close'] >= df_copy.loc[last_row_index, 'SMA_224'] * (1 - ma_224_tolerance)) & \
                                                       (df_copy.loc[last_row_index, 'Close'] <= df_copy.loc[last_row_index, 'SMA_224'] * (1 + ma_224_tolerance)) & \
                                                       ((df_copy.loc[last_row_index, 'Close'] > df_copy.loc[last_row_index, 'SMA_224']) | (df_copy.loc[last_row_index, 'Low'] <= df_copy.loc[last_row_index, 'SMA_224']))).astype(int)

    volume_increase_ratio = 1.2
    features.loc[last_row_index, 'Volume_Condition'] = ((df_copy.loc[last_row_index, 'VMA'] > 0) & (df_copy.loc[last_row_index, 'Volume'] >= df_copy.loc[last_row_index, 'VMA'] * volume_increase_ratio)).astype(int)

    features.dropna(inplace=True)

    # AI 모델 학습 시 사용된 특징 순서와 동일하게 유지
    expected_features_order = [
        'RSI', 'STOCH_K', 'STOCH_D', 'DMI_ADX', 'DMI_Plus_Minus_Diff', 'Close_vs_SAR_Diff',
        'Close_vs_SMA224_Ratio', 'Volume_vs_VMA_Ratio', 'RSI_Change_5d', 'STOCH_K_Change_5d',
        'Close_Change_5d', 'SAR_Buy_Trend', 'RSI_GC', 'RSI_In_Buy_Zone',
        'RSI_Not_High_Downtrend', 'STOCH_GC', 'DMI_Condition', 'MA224_Condition', 'Volume_Condition'
    ]

    if not features.empty and all(col in features.columns for col in expected_features_order):
        return features[expected_features_order].iloc[0].to_frame().T
    return pd.DataFrame()

# --- 핵심 백테스팅 로직 함수 ---
def run_backtest(stock_data_df,
                 rsi_buy_zone_min=0, rsi_buy_zone_max=45,
                 rsi_ma_short_window=5, rsi_ma_long_window=20,
                 rsi_overbought_zone_min=70,
                 stoch_k_window=14, stoch_d_window=3,
                 dmi_window=14, dmi_adx_min=15,
                 ma_224_tolerance_percent=0.1,
                 ichimoku_tenkan=9, ichimoku_kijun=26, ichimoku_senkou=52,
                 volume_ma_window=20,
                 volume_increase_ratio=1.2,
                 holding_periods=[5, 10, 20],
                 condition_lookback_window=5,
                 ai_prediction_threshold=0.85 # AI 보조 신호 활성화 임계치를 0.85 (85%)로 변경
                ):

    df = stock_data_df.copy()
    df.rename(columns={
        'open_price': 'Open',
        'high_price': 'High',
        'low_price': 'Low',
        'close_price': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    df.set_index('date', inplace=True)
    df.sort_index(ascending=True, inplace=True)

    # --- 1. 기술적 지표 계산 ---
    df['SAR'] = calculate_psar_manual(df)

    # PSAR 매수 전환 신호: (점(SAR)이 캔들 위에 있다가 캔들 아래로 전환)
    df['PSAR_Reversal_Buy_Current'] = (df['Close'] > df['SAR']) & \
                                     (df['Close'].shift(1).notna()) & (df['SAR'].shift(1).notna()) & \
                                     (df['Close'].shift(1) <= df['SAR'].shift(1))

    df['RSI'] = ta.momentum.rsi(df['Close'], window=14, fillna=False)
    df['RSI_SMA_Short'] = ta.trend.sma_indicator(df['RSI'], window=rsi_ma_short_window, fillna=False)
    df['RSI_SMA_Long'] = ta.trend.sma_indicator(df['RSI'], window=rsi_ma_long_window, fillna=False)

    df['RSI_GC_Current'] = (df['RSI_SMA_Short'] > df['RSI_SMA_Long']) & \
                           (df['RSI_SMA_Short'].shift(1) <= df['RSI_SMA_Long'].shift(1))

    df['RSI_In_Buy_Zone_Current'] = (df['RSI'] >= rsi_buy_zone_min) & \
                                    (df['RSI'] <= rsi_buy_zone_max)

    df['RSI_Downtrend_Current'] = False
    df.loc[(df['RSI_SMA_Short'] < df['RSI_SMA_Short'].shift(1)) &
           (df['RSI'] >= rsi_overbought_zone_min), 'RSI_Downtrend_Current'] = True

    df['STOCH_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=stoch_k_window, fillna=False)
    df['STOCH_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=stoch_k_window, smooth_window=stoch_d_window, fillna=False)
    df['STOCH_GC_Current'] = (df['STOCH_K'] > df['STOCH_D']) & \
                             (df['STOCH_K'].shift(1) <= df['STOCH_D'].shift(1))

    adx_indicator = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=dmi_window, fillna=False)
    df['DMI_plus_di'] = adx_indicator.adx_pos()
    df['DMI_minus_di'] = adx_indicator.adx_neg()
    df['DMI_adx'] = adx_indicator.adx()
    df['DMI_Condition_Current'] = (df['DMI_plus_di'] > df['DMI_minus_di']) & \
                                  (df['DMI_adx'] >= dmi_adx_min)

    # 일목균형표는 현재 매수 전략에 직접 사용되지는 않지만, 나중에 확장할 수 있도록 계산
    ichimoku_indicator = ta.trend.IchimokuIndicator(
        high=df['High'], low=df['Low'], window1=ichimoku_tenkan, window2=ichimoku_kijun, window3=ichimoku_senkou, fillna=False
    )
    df['ICHIMOKU_TENKAN_SEN'] = ichimoku_indicator.ichimoku_conversion_line()
    df['ICHIMOKU_KIJUN_SEN'] = ichimoku_indicator.ichimoku_base_line()
    df['ICHIMOKU_SENKOU_SPAN_A'] = ichimoku_indicator.ichimoku_a()
    df['ICHIMOKU_SENKOU_SPAN_B'] = ichimoku_indicator.ichimoku_b()
    df['ICHIMOKU_CLOUD_TOP'] = df[['ICHIMOKU_SENKOU_SPAN_A', 'ICHIMOKU_SENKOU_SPAN_B']].max(axis=1)
    df['ICHIMOKU_CLOUD_BOTTOM'] = df[['ICHIMOKU_SENKOU_SPAN_A', 'ICHIMOKU_SENKOU_SPAN_B']].min(axis=1)
    df['ICHIMOKU_Positive_Cloud_Current'] = df['ICHIMOKU_SENKOU_SPAN_A'] > df['ICHIMOKU_SENKOU_SPAN_B']
    df['ICHIMOKU_Cloud_Support_Current'] = (df['Close'] >= df['ICHIMOKU_CLOUD_BOTTOM']) | \
                                           (df['Close'] > df['ICHIMOKU_CLOUD_BOTTOM'].shift(1)) & (df['Close'].shift(1) <= df['ICHIMOKU_CLOUD_BOTTOM'].shift(1))

    df['SMA_224'] = ta.trend.sma_indicator(df['Close'], window=224, fillna=False)
    df['MA_224_Condition_Current'] = (df['Close'] >= df['SMA_224'] * (1 - ma_224_tolerance_percent)) & \
                                     (df['Close'] <= df['SMA_224'] * (1 + ma_224_tolerance_percent)) & \
                                     ((df['Close'] > df['SMA_224']) | (df['Low'] <= df['SMA_224']))

    df['VMA'] = df['Volume'].rolling(window=volume_ma_window, min_periods=1).mean()
    df['Volume_Condition_Current'] = (df['VMA'] > 0) & (df['Volume'] >= df['VMA'] * volume_increase_ratio)

    # --- 각 조건별 '최근 N일 내 만족 여부' 플래그 생성 ---
    # `.astype(bool)`을 사용하여 NaN이 아닌 유효한 값만 처리
    df['PSAR_Reversal_Buy_Recently_Met'] = df['PSAR_Reversal_Buy_Current'].rolling(window=condition_lookback_window, min_periods=1).max().astype(bool)

    df['RSI_GC_Recently_Met'] = df['RSI_GC_Current'].rolling(window=condition_lookback_window, min_periods=1).max().astype(bool)
    df['RSI_In_Buy_Zone_Recently_Met'] = df['RSI_In_Buy_Zone_Current'].rolling(window=condition_lookback_window, min_periods=1).max().astype(bool)
    df['STOCH_GC_Recently_Met'] = df['STOCH_GC_Current'].rolling(window=condition_lookback_window, min_periods=1).max().astype(bool)
    df['DMI_Condition_Recently_Met'] = df['DMI_Condition_Current'].rolling(window=condition_lookback_window, min_periods=1).max().astype(bool)
    df['MA_224_Condition_Recently_Met'] = df['MA_224_Condition_Current'].rolling(window=condition_lookback_window, min_periods=1).max().astype(bool)
    df['Volume_Condition_Recently_Met'] = df['Volume_Condition_Current'].rolling(window=condition_lookback_window, min_periods=1).max().astype(bool)

    # AI 모델 특징 계산에 필요한 최소 데이터
    # SMA_224 계산에 224일, RSI_Change_5d 등에 5일 필요, SAR도 2일 필요 등
    # 가장 긴 기간인 224일 SMA를 고려하여 min_periods 조정
    min_data_for_ai = 224 + 5 # 224일 SMA + 5일 변화량 계산에 충분한 기간

    # 초기 NaN 값 제거 (지표 계산 후 유효한 데이터만 남김)
    # 필요한 모든 지표들이 계산된 후에 유효한 행들만 남도록 dropna를 수행
    required_initial_cols = ['RSI', 'STOCH_K', 'DMI_adx', 'SMA_224', 'VMA', 'SAR',
                             'PSAR_Reversal_Buy_Recently_Met', 'RSI_GC_Recently_Met', 'RSI_In_Buy_Zone_Recently_Met',
                             'STOCH_GC_Recently_Met', 'DMI_Condition_Recently_Met', 'MA_224_Condition_Recently_Met',
                             'Volume_Condition_Recently_Met']
    df.dropna(subset=required_initial_cols, inplace=True)


    if df.empty:
        return {"error": "데이터 부족: 요청 기간에 필요한 최소한의 데이터가 없거나, 지표 계산 후 유효한 데이터가 충분하지 않습니다."}

    # --- 2. 백테스팅 시뮬레이션 및 과거 매수 신호 기록 ---
    trading_results = []
    past_buy_dates = [] # 과거 매수 신호 발생 날짜를 저장
    
    # 백테스트 시작 인덱스 조정 (AI 모델에 필요한 최소 데이터 기간 이후부터)
    start_backtest_idx = 0
    if len(df) > min_data_for_ai:
        start_backtest_idx = df.index.get_loc(df.iloc[min_data_for_ai - 1].name) + 1 # 최소 데이터 이후부터 시작

    for i in range(start_backtest_idx, len(df)):
        current_date = df.index[i]
        current_row = df.loc[current_date]

        # RSI 하락 추세일 경우 매수 신호 검사 중단 (우선 순위가 높음)
        if current_row['RSI_Downtrend_Current']:
            continue

        # RSI 복합 조건: GC 또는 매수 영역 진입
        rsi_combined_condition = bool(current_row['RSI_GC_Recently_Met']) or \
                                 bool(current_row['RSI_In_Buy_Zone_Recently_Met'])

        # 주요 매수 조건 (모두 만족해야 함)
        primary_buy_signal = (bool(current_row['PSAR_Reversal_Buy_Recently_Met']) and
                              rsi_combined_condition and
                              bool(current_row['STOCH_GC_Recently_Met']) and
                              bool(current_row['DMI_Condition_Recently_Met']) and
                              bool(current_row['MA_224_Condition_Recently_Met']) and
                              bool(current_row['Volume_Condition_Recently_Met']))

        ai_supplemental_signal = False
        ai_prediction_score = None # 0 대신 None으로 시작 (UI에서 '-'로 표시되도록)

        # AI 모델이 로드되었고, 현재 데이터가 AI 예측에 필요한 최소 조건을 만족하며
        # 주요 매수 신호가 발생하지 않았을 경우에만 AI 보조 신호 검사
        if ai_model is not None and len(df.iloc[:i+1]) >= min_data_for_ai:
            if not primary_buy_signal: # 기본 신호(절대지표)가 아닐 때만 AI 보조 신호 확인
                # AI 예측을 위해 현재 날짜까지의 데이터만 전달
                current_df_segment = df.iloc[:i+1]
                ai_features_df = extract_features_for_ai_prediction(current_df_segment)

                if not ai_features_df.empty:
                    try:
                        # AI 모델 예측 수행
                        ai_prediction_prob = ai_model.predict_proba(ai_features_df)[0][1]
                        ai_prediction_score = float(round(ai_prediction_prob * 100, 2))
                        if ai_prediction_score == 0: # 0점일 경우 None으로 처리하여 UI에서 '-'로 표시
                            ai_prediction_score = None

                        # AI_Prediction_Score_Percent 조건 추가: 85점 이상일 때만 AI 보조 신호 활성화
                        if ai_prediction_prob >= ai_prediction_threshold:
                            ai_supplemental_signal = True
                    except Exception as e:
                        print(f"AI 모델 예측 오류 (날짜: {current_date}): {e}")

        # 최종 매수 조건: 주요 매수 신호 발생 OR AI 보조 신호 발생
        if (primary_buy_signal or ai_supplemental_signal):

            past_buy_dates.append(current_date.strftime('%Y-%m-%d'))

            entry_details = {
                'entry_date': current_date.strftime('%Y-%m-%d'),
                'entry_price': float(current_row['Close']),
                'signals': { # 매수 신호 발생 당시의 상세 조건 기록
                    'Primary_Conditions_Met': bool(primary_buy_signal), # 절대지표 만족 여부
                    'AI_Supplemental_Signal': bool(ai_supplemental_signal), # AI 지표 만족 여부
                    'AI_Prediction_Score_Percent': ai_prediction_score,
                    'PSAR_Reversal_Buy_Recently_Met': bool(current_row['PSAR_Reversal_Buy_Recently_Met']),
                    'RSI_GC_Recently_Met': bool(current_row['RSI_GC_Recently_Met']),
                    'RSI_In_Buy_Zone_Recently_Met': bool(current_row['RSI_In_Buy_Zone_Recently_Met']),
                    'STOCH_GC_Recently_Met': bool(current_row['STOCH_GC_Recently_Met']),
                    'DMI_Condition_Recently_Met': bool(current_row['DMI_Condition_Recently_Met']),
                    'MA_224_Condition_Recently_Met': bool(current_row['MA_224_Condition_Recently_Met']),
                    'Volume_Condition_Recently_Met': bool(current_row['Volume_Condition_Recently_Met']),
                    'RSI_Condition_Met_Overall': bool(rsi_combined_condition), # RSI 복합 조건 만족 여부
                    'RSI_Current': float(round(current_row['RSI'], 2)) if pd.notna(current_row['RSI']) else None,
                    'STOCH_K_Current': float(round(current_row['STOCH_K'], 2)) if pd.notna(current_row['STOCH_K']) else None,
                    'DMI_ADX_Current': float(round(current_row['DMI_adx'], 2)) if pd.notna(current_row['DMI_adx']) else None,
                    'SMA_224_Current': float(round(current_row['SMA_224'], 2)) if pd.notna(current_row['SMA_224']) else None,
                    'Volume_Ratio_Current': float(round(current_row['Volume'] / current_row['VMA'], 2)) if current_row['VMA'] > 0 and pd.notna(current_row['Volume']) and pd.notna(current_row['VMA']) else None
                },
                'signal_type': 'AI' if ai_supplemental_signal else '절대지표' # 신호 타입 추가
            }

            # 각 보유 기간별 수익률 계산 (이 부분은 수정 없음)
            for holding_period in holding_periods:
                exit_date_index = i + holding_period
                if exit_date_index < len(df): # 데이터 범위 내에 매도일이 존재하면
                    exit_row = df.iloc[exit_date_index]
                    exit_price = float(exit_row['Close'])
                    profit_loss_percent = ((exit_price - entry_details['entry_price']) / entry_details['entry_price']) * 100

                    result = entry_details.copy() # 매수 정보 복사
                    result['exit_date'] = exit_row.name.strftime('%Y-%m-%d')
                    result['exit_price'] = exit_price
                    result[f'profit_loss_percent_{holding_period}d'] = float(round(profit_loss_percent, 2))
                    trading_results.append(result)
                else: # 데이터 부족으로 매도일 계산 불가
                    result = entry_details.copy()
                    result['exit_date'] = None # 'N/A' 대신 None
                    result['exit_price'] = None # 'N/A' 대신 None
                    result[f'profit_loss_percent_{holding_period}d'] = None # 'N/A (데이터 부족)' 대신 None
                    trading_results.append(result)

    # --- 현재 날짜의 매수 신호 여부 확인 및 조건부 예상 보유 기간 수익률 추가 ---
    current_buy_signal = {
        'is_buy_signal': False,
        'signals': {},
        'predicted_holding_period_returns': None # 초기값 None
    }

    # 최신 데이터에 대한 매수 신호 검사 (차트 표시 및 현재 상황 분석용)
    # 마지막 날짜가 AI 모델 계산에 필요한 최소 데이터 기간 이후부터
    latest_buy_signal_date_for_chart = None # 기본값 None
    if past_buy_dates:
        latest_buy_signal_date_for_chart = past_buy_dates[-1] # 백테스팅된 가장 최근 매수일

    if not df.empty and len(df) >= min_data_for_ai:
        latest_date = df.index[-1]
        latest_row = df.loc[latest_date]

        # 최신 행에 대해 필요한 모든 컬럼이 NaN이 아닌지 확인
        required_cols_for_check_latest = [
            'PSAR_Reversal_Buy_Recently_Met',
            'RSI_GC_Recently_Met', 'RSI_In_Buy_Zone_Recently_Met',
            'STOCH_GC_Recently_Met', 'DMI_Condition_Recently_Met', 'MA_224_Condition_Recently_Met',
            'Volume_Condition_Recently_Met', 'RSI_Downtrend_Current'
        ]
        if not any(pd.isna(latest_row[col]) for col in required_cols_for_check_latest):
            if not latest_row['RSI_Downtrend_Current']: # RSI 하락 추세가 아닐 때만 검사
                rsi_combined_condition_latest = bool(latest_row['RSI_GC_Recently_Met']) or \
                                                bool(latest_row['RSI_In_Buy_Zone_Recently_Met'])

                primary_buy_signal_latest = (bool(latest_row['PSAR_Reversal_Buy_Recently_Met']) and
                                             rsi_combined_condition_latest and
                                             bool(latest_row['STOCH_GC_Recently_Met']) and
                                             bool(latest_row['DMI_Condition_Recently_Met']) and
                                             bool(latest_row['MA_224_Condition_Recently_Met']) and
                                             bool(latest_row['Volume_Condition_Recently_Met']))

                ai_supplemental_signal_latest = False
                ai_prediction_score_latest = None # 0 대신 None으로 시작

                # AI 모델이 로드되었고 충분한 데이터가 있을 때만 AI 예측 시도
                if ai_model is not None:
                    current_df_segment_latest = df.iloc[:] # 전체 데이터프레임을 AI 예측 함수에 전달
                    ai_features_df_latest = extract_features_for_ai_prediction(current_df_segment_latest)

                    if not ai_features_df_latest.empty:
                        try:
                            ai_prediction_prob_latest = ai_model.predict_proba(ai_features_df_latest)[0][1]
                            ai_prediction_score_latest = float(round(ai_prediction_prob_latest * 100, 2))
                            if ai_prediction_score_latest == 0: # 0점일 경우 None으로 처리하여 UI에서 '-'로 표시
                                ai_prediction_score_latest = None

                            if ai_prediction_prob_latest >= ai_prediction_threshold:
                                ai_supplemental_signal_latest = True
                        except Exception as e:
                            print(f"AI 모델 예측 오류 (최신 날짜: {latest_date}): {e}")

                # 최신 날짜에 매수 신호 발생 여부 결정
                if primary_buy_signal_latest or ai_supplemental_signal_latest:
                    current_buy_signal['is_buy_signal'] = True
                    current_buy_signal['signals'] = {
                        'Primary_Conditions_Met': bool(primary_buy_signal_latest),
                        'AI_Supplemental_Signal': bool(ai_supplemental_signal_latest),
                        'AI_Prediction_Score_Percent': ai_prediction_score_latest,
                        'PSAR_Reversal_Buy_Recently_Met': bool(latest_row['PSAR_Reversal_Buy_Recently_Met']),
                        'RSI_GC_Recently_Met': bool(latest_row['RSI_GC_Recently_Met']),
                        'RSI_In_Buy_Zone_Recently_Met': bool(latest_row['RSI_In_Buy_Zone_Recently_Met']),
                        'STOCH_GC_Recently_Met': bool(latest_row['STOCH_GC_Recently_Met']),
                        'DMI_Condition_Recently_Met': bool(latest_row['DMI_Condition_Recently_Met']),
                        'MA_224_Condition_Recently_Met': bool(latest_row['MA_224_Condition_Recently_Met']),
                        'Volume_Condition_Recently_Met': bool(latest_row['Volume_Condition_Recently_Met']),
                        'RSI_Condition_Met_Overall': bool(rsi_combined_condition_latest),
                        'RSI_Current': float(round(latest_row['RSI'], 2)) if pd.notna(latest_row['RSI']) else None,
                        'STOCH_K_Current': float(round(latest_row['STOCH_K'], 2)) if pd.notna(latest_row['STOCH_K']) else None,
                        'DMI_ADX_Current': float(round(latest_row['DMI_adx'], 2)) if pd.notna(latest_row['DMI_adx']) else None,
                        'SMA_224_Current': float(round(latest_row['SMA_224'], 2)) if pd.notna(latest_row['SMA_224']) else None,
                        'Volume_Ratio_Current': float(round(latest_row['Volume'] / latest_row['VMA'], 2)) if latest_row['VMA'] > 0 and pd.notna(latest_row['Volume']) and pd.notna(latest_row['VMA']) else None
                    }

                    # 현재 신호에 대한 예상 수익률은 전체 과거 거래의 평균으로 제공
                    all_holding_period_profits = {hp: [] for hp in holding_periods}
                    for trade in trading_results:
                        for hp in holding_periods:
                            key = f'profit_loss_percent_{hp}d'
                            if key in trade and isinstance(trade[key], (int, float)):
                                all_holding_period_profits[hp].append(trade[key])

                    predicted_returns_for_current_signal = {}
                    for hp in holding_periods:
                        if all_holding_period_profits[hp]:
                            predicted_returns_for_current_signal[f'{hp}d'] = float(round(np.mean(all_holding_period_profits[hp]), 2))
                        else:
                            predicted_returns_for_current_signal[f'{hp}d'] = None
                    
                    current_buy_signal['predicted_holding_period_returns'] = predicted_returns_for_current_signal


    return {
        "trading_results": trading_results,
        "past_buy_dates": past_buy_dates,
        "latest_buy_signal_date_for_chart": latest_buy_signal_date_for_chart,
        "current_buy_signal": current_buy_signal,
        "full_df_for_chart": df.reset_index().to_dict('records')
    }


def calculate_profit_loss_statistics(trades, holding_periods=[5, 10, 20], signal_type='overall'):
    """
    주어진 거래 목록과 보유 기간에 대해 수익률 통계를 계산합니다.
    signal_type: 'overall', 'AI', '절대지표' 중 하나.
    """
    filtered_trades = []
    if signal_type == 'overall':
        filtered_trades = trades
    else:
        filtered_trades = [trade for trade in trades if trade.get('signal_type') == signal_type]

    all_profit_losses = []
    for trade in filtered_trades:
        for hp in holding_periods:
            key = f'profit_loss_percent_{hp}d'
            if key in trade and isinstance(trade[key], (int, float)):
                all_profit_losses.append(trade[key])

    profit_loss_stats = {}
    profit_loss_distribution = {}

    if all_profit_losses:
        all_profit_losses_np = np.array(all_profit_losses)

        profit_loss_stats = {
            'max': float(round(np.max(all_profit_losses_np), 2)),
            'min': float(round(np.min(all_profit_losses_np), 2)),
            'average': float(round(np.mean(all_profit_losses_np), 2)),
            'median': float(round(np.median(all_profit_losses_np), 2))
        }

        # 백분위수 기반 통계
        q25 = np.percentile(all_profit_losses_np, 25)
        q75 = np.percentile(all_profit_losses_np, 75)

        bottom_25_percent = all_profit_losses_np[all_profit_losses_np <= q25]
        if len(bottom_25_percent) > 0:
            profit_loss_stats['bottom_25_percent'] = {
                'max': float(round(np.max(bottom_25_percent), 2)),
                'min': float(round(np.min(bottom_25_percent), 2)),
                'average': float(round(np.mean(bottom_25_percent), 2)),
                'median': float(round(np.median(bottom_25_percent), 2))
            }
        else:
            profit_loss_stats['bottom_25_percent'] = None
        
        top_25_percent = all_profit_losses_np[all_profit_losses_np >= q75]
        if len(top_25_percent) > 0:
            profit_loss_stats['top_25_percent'] = {
                'max': float(round(np.max(top_25_percent), 2)),
                'min': float(round(np.min(top_25_percent), 2)),
                'average': float(round(np.mean(top_25_percent), 2)),
                'median': float(round(np.median(top_25_percent), 2))
            }
        else:
            profit_loss_stats['top_25_percent'] = None

        # 분포 계산
        bins = [-np.inf, -10, 0, 5, 10, np.inf]
        labels = ['< -10%', '-10% ~ 0%', '0% ~ 5%', '5% ~ 10%', '> 10%']

        cut_results = pd.cut(all_profit_losses_np, bins=bins, labels=labels, right=True, include_lowest=True)
        distribution_counts = cut_results.value_counts().sort_index()
        total_trades_count = len(all_profit_losses_np)

        if total_trades_count > 0:
            for label in labels:
                count = distribution_counts.get(label, 0)
                probability = float(round((count / total_trades_count) * 100, 2)) if total_trades_count > 0 else 0.0
                profit_loss_distribution[label] = {
                    'count': int(count),
                    'probability_percent': probability
                }
        else:
            profit_loss_distribution = {label: {'count': 0, 'probability_percent': 0.0} for label in labels}

    else:
        profit_loss_stats = None
        profit_loss_distribution = {label: {'count': 0, 'probability_percent': 0.0} for label in labels}


    return profit_loss_stats, profit_loss_distribution


@require_GET
def analyze_stock_strategy(request):
    symbol = request.GET.get('symbol')
    start_date_str = request.GET.get('start_date')
    end_date_str = request.GET.get('end_date')

    if not symbol:
        return JsonResponse({'error': '종목 코드(symbol)가 필요합니다.'}, status=400)

    if not end_date_str:
        end_date = date.today()
        end_date_str = end_date.strftime('%Y-%m-%d')
    else:
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

    if not start_date_str:
        start_date = end_date - timedelta(days=365 * 4) # 4년치 데이터
        start_date_str = start_date.strftime('%Y-%m-%d')
    else:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()

    try:
        stock = Stock.objects.get(symbol=symbol)
        print(f"'{symbol}' 종목 데이터베이스에 존재. 백테스팅을 진행합니다.")
    except Stock.DoesNotExist:
        print(f"'{symbol}' 종목 데이터베이스에 없음. 데이터 수집을 시도합니다.")
        try:
            from .collect_data import collect_stock_data
            collect_result = collect_stock_data(symbol)
            if 'error' in collect_result:
                return JsonResponse({'error': f'종목 정보가 없어 데이터를 수집하려 했으나 오류 발생: {collect_result["error"]}'}, status=500)
            stock = Stock.objects.get(symbol=symbol)
            print(f"'{symbol}' 종목 데이터 수집 완료. 백테스팅을 진행합니다.")
        except ImportError:
             return JsonResponse({'error': '데이터 수집 모듈이 없어 종목을 찾을 수 없습니다. collect_data.py 파일이 올바르게 존재하는지 확인하세요.'}, status=500)
        except Exception as e:
            return JsonResponse({'error': f'종목 데이터 자동 수집 중 오류 발생: {str(e)}'}, status=500)

    daily_prices_queryset = StockDailyPrice.objects.filter(stock=stock).order_by('date')
    daily_prices_queryset = daily_prices_queryset.filter(date__gte=start_date_str, date__lte=end_date_str)

    data_list = list(daily_prices_queryset.values(
        'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume'
    ))

    if not data_list:
        return JsonResponse({'error': f'요청하신 기간 ({start_date_str} ~ {end_date_str})에 "{symbol}" 종목의 주가 데이터가 없습니다. 기간을 조정하거나 최신 데이터 수집을 확인해주세요.'}, status=404)

    df = pd.DataFrame(data_list)

    try:
        backtest_output = run_backtest(df.copy())

        if isinstance(backtest_output, dict) and 'error' in backtest_output:
            return JsonResponse(backtest_output, status=400)

        trading_results = backtest_output['trading_results']
        past_buy_dates = backtest_output['past_buy_dates']
        latest_buy_signal_date_for_chart = backtest_output['latest_buy_signal_date_for_chart']
        current_buy_signal = backtest_output['current_buy_signal']
        full_df_for_chart = backtest_output['full_df_for_chart']

        # 차트 시각화를 위한 데이터 준비
        chart_data = []
        for row_dict in full_df_for_chart:
            chart_data.append({
                'date': row_dict['date'].strftime('%Y-%m-%d') if isinstance(row_dict['date'], date) else row_dict['date'],
                'open': float(row_dict['Open']),
                'high': float(row_dict['High']),
                'low': float(row_dict['Low']),
                'close': float(row_dict['Close']),
                'sar': float(row_dict['SAR']) if 'SAR' in row_dict and pd.notna(row_dict['SAR']) else None,
                'isBuySignalPoint': row_dict['date'].strftime('%Y-%m-%d') in past_buy_dates
            })

        # --- 수익률 통계 계산 (종합, AI 지표, 절대 지표) ---
        profit_loss_stats_overall, profit_loss_distribution_overall = calculate_profit_loss_statistics(trading_results, holding_periods=[5, 10, 20], signal_type='overall')
        profit_loss_stats_ai, profit_loss_distribution_ai = calculate_profit_loss_statistics(trading_results, holding_periods=[5, 10, 20], signal_type='AI')
        profit_loss_stats_absolute, profit_loss_distribution_absolute = calculate_profit_loss_statistics(trading_results, holding_periods=[5, 10, 20], signal_type='절대지표')


        return JsonResponse({
            'symbol': symbol,
            'current_buy_signal': current_buy_signal,
            'past_buy_dates': past_buy_dates,
            'latest_buy_signal_date_for_chart': latest_buy_signal_date_for_chart,
            'profit_loss_statistics': {
                'overall': profit_loss_stats_overall,
                'ai_signal': profit_loss_stats_ai,
                'absolute_signal': profit_loss_stats_absolute,
            },
            'profit_loss_distribution': {
                'overall': profit_loss_distribution_overall,
                'ai_signal': profit_loss_distribution_ai,
                'absolute_signal': profit_loss_distribution_absolute,
            },
            'backtest_trades': trading_results,
            'chart_data': chart_data,
        })
    except Exception as e:
        return JsonResponse({'error': f'백테스팅 및 분석 중 오류 발생: {str(e)}'}, status=500)


@require_GET
def stock_analysis_page(request):
    """
    주식 분석 웹 페이지를 렌더링합니다.
    사용자가 종목 코드와 기간을 입력할 수 있는 폼과 결과를 표시할 공간을 제공합니다.
    """
    return render(request, 'stock_analysis_page.html')