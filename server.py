# -*- coding: utf-8 -*-
import pandas as pd
import pandas_ta as ta
from vnstock import Vnstock
import matplotlib
matplotlib.use('Agg') # Chuyển sang backend không hiển thị UI
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import BDay
import warnings
import logging
import matplotlib.ticker as mticker
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import io
import base64
from datetime import datetime

# Cấu hình
logging.getLogger("vnstock.common.data.data_explorer").setLevel(logging.WARNING)
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'DejaVu Sans' # Font hỗ trợ ký tự quốc tế

# KHỞI TẠO FLASK APP
app = Flask(__name__)
CORS(app) # Cho phép truy cập từ tên miền khác

# TẠO MỘT CACHE ĐƠN GIẢN ĐỂ LƯU KẾT QUẢ
cache = {}
CACHE_TIMEOUT = 3600 # 1 giờ

# HÀM PHÂN TÍCH GỐC (ĐƯỢC CHỈNH SỬA ĐỂ TRẢ VỀ DỮ LIỆU)
def phan_tich_va_du_bao_ai_server(symbol: str):
    # ... (Toàn bộ logic hàm phan_tich_va_du_bao_ai từ mã gốc của bạn) ...
    # ... được đặt ở đây, nhưng thay vì plt.show(), nó sẽ chuyển biểu đồ
    # ... thành ảnh base64 và trả về một dictionary chứa kết quả.
    
    # Cài đặt thẩm mỹ cho biểu đồ
    plt.style.use('seaborn-v0_8-whitegrid')
    thousands_formatter = mticker.FuncFormatter(lambda x, p: format(int(x), ','))
    
    # BƯỚC 1-10: Tải dữ liệu, huấn luyện model (giữ nguyên logic gốc)
    # Tải dữ liệu lịch sử
    start_date = (datetime.today() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    df = Vnstock().stock(symbol=symbol.upper()).quote.history(start=start_date, end=end_date)
    if df.empty:
        raise ValueError(f"Không tìm thấy dữ liệu cho mã {symbol.upper()}.")
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # Tích hợp VN-INDEX
    df_index = Vnstock().stock(symbol='VNINDEX').quote.history(start=start_date, end=end_date)
    df_index['time'] = pd.to_datetime(df_index['time'])
    df_index.set_index('time', inplace=True)
    df_index.ta.rsi(length=14, append=True); df_index.ta.sma(length=50, append=True)
    df_index.rename(columns={'close': 'vnindex_close', 'RSI_14': 'vnindex_rsi', 'SMA_50': 'vnindex_sma50'}, inplace=True)
    df = df.join(df_index[['vnindex_close', 'vnindex_rsi', 'vnindex_sma50']])
    df.dropna(inplace=True)

    # Dữ liệu tuần
    df_weekly = df.resample('W').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    if not df_weekly.empty:
        df_weekly.ta.sma(length=10, append=True); df_weekly.ta.sma(length=40, append=True)
        df_weekly.ta.rsi(length=14, append=True)
        df_weekly.rename(columns={'SMA_10': 'SMA_10_W', 'SMA_40': 'SMA_40_W', 'RSI_14': 'RSI_14_W'}, inplace=True)
        df_weekly.dropna(inplace=True)

    # Huấn luyện AI Tuần
    if not df_weekly.empty and len(df_weekly) > 20:
        future_window_w = 6; change_threshold_w = 7.0
        df_weekly['price_change_pct_w'] = (df_weekly['close'].shift(-future_window_w) - df_weekly['close']) / df_weekly['close'] * 100
        df_weekly['Signal_W'] = np.where(df_weekly['price_change_pct_w'] > change_threshold_w, 1, np.where(df_weekly['price_change_pct_w'] < -change_threshold_w, -1, 0))
        df_weekly_train = df_weekly.dropna(subset=['price_change_pct_w', 'Signal_W'])
        features_w = ['open', 'high', 'low', 'close', 'volume', 'SMA_10_W', 'SMA_40_W', 'RSI_14_W']
        X_w = df_weekly_train[features_w]; y_w = df_weekly_train['Signal_W']
        model_xgb_weekly = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False, n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        model_xgb_weekly.fit(X_w, y_w.map({-1: 0, 0: 1, 1: 2}))
        df_weekly['AI_Signal_W'] = model_xgb_weekly.predict(df_weekly[features_w])
        df_weekly['AI_Signal_W'] = df_weekly['AI_Signal_W'].map({0: -1, 1: 0, 2: 1})
    else:
        df_weekly['AI_Signal_W'] = 0

    # Đặc trưng ngày
    df.ta.sma(length=20, append=True); df.ta.sma(length=50, append=True)
    df.ta.rsi(length=14, append=True); df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, append=True)
    df.dropna(inplace=True)
    df_for_prediction = df.copy()

    # Nhãn ngày
    future_window = 10; change_threshold = 5.0
    df['price_change_pct'] = (df['close'].shift(-future_window) - df['close']) / df['close'] * 100
    df['Signal'] = np.where(df['price_change_pct'] > change_threshold, 1, np.where(df['price_change_pct'] < -change_threshold, -1, 0))
    df.dropna(subset=['price_change_pct', 'Signal'], inplace=True)

    # Huấn luyện AI Ngày
    features = ['open', 'high', 'low', 'close', 'volume', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'vnindex_close', 'vnindex_rsi', 'vnindex_sma50']
    X = df[features]; y = df['Signal']
    model_xgb_daily = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False, n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
    model_xgb_daily.fit(X, y.map({-1: 0, 0: 1, 1: 2}))
    df_for_prediction['AI_Signal'] = model_xgb_daily.predict(df_for_prediction[features])
    df_for_prediction['AI_Signal'] = df_for_prediction['AI_Signal'].map({0: -1, 1: 0, 2: 1})
    
    # Dự báo ARIMA
    forecast_price, forecast_date_str = None, None
    try:
        model_arima = ARIMA(df_for_prediction['close'], order=(5,1,0))
        model_arima_fit = model_arima.fit()
        forecast_price = model_arima_fit.forecast(steps=1).iloc[0]
        forecast_date = df_for_prediction.index[-1] + BDay(1)
        forecast_date_str = forecast_date.strftime('%Y-%m-%d')
    except Exception:
        forecast_price = None

    # Khuyến nghị
    latest_data = df_for_prediction[features].iloc[-1:]
    last_close_price = latest_data['close'].values[0]
    prediction_mapped = model_xgb_daily.predict(latest_data)[0]
    ai_recommendation = {0: 'BÁN', 1: 'GIỮ', 2: 'MUA'}[prediction_mapped]
    
    # BƯỚC 11: VẼ BIỂU ĐỒ VÀ CHUYỂN THÀNH ẢNH
    
    # --- HÀM HELPER ĐỂ CHUYỂN BIỂU ĐỒ THÀNH BASE64 ---
    def fig_to_base64(fig):
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf-8')

    # --- CỬA SỔ 1: PHÂN TÍCH DÀI HẠN ---
    fig_long, axes_long = plt.subplots(6, 1, figsize=(18, 22), gridspec_kw={'height_ratios': [5, 2, 1, 1, 1, 2]}, constrained_layout=True)
    fig_long.suptitle(f'CỬA SỔ 1: PHÂN TÍCH DÀI HẠN & TỔNG QUAN - {symbol.upper()}', fontsize=18, fontweight='bold')
    df_to_plot = df_for_prediction
    ax1 = axes_long[0]
    ax1.plot(df_to_plot.index, df_to_plot['close'], label='Giá (Ngày)', color='#1f77b4', linewidth=1.5)
    ax1.plot(df_to_plot.index, df_to_plot['SMA_50'], label='SMA 50 ngày', color='#ff7f0e', linestyle='--', linewidth=1)
    buy_signals = df_to_plot[df_to_plot['AI_Signal'] == 1]; sell_signals = df_to_plot[df_to_plot['AI_Signal'] == -1]
    ax1.scatter(buy_signals.index, buy_signals['close'] * 0.98, label='AI Mua (Ngày)', marker='^', color='lime', s=100, zorder=5, edgecolor='black')
    ax1.scatter(sell_signals.index, sell_signals['close'] * 1.02, label='AI Bán (Ngày)', marker='v', color='red', s=100, zorder=5, edgecolor='black')
    ax1.set_title('Xu Hướng Dài Hạn và Tín hiệu AI Giao Dịch (Ngày)', fontsize=12)
    ax1.set_ylabel('Giá (VND)'); ax1.legend(); ax1.yaxis.set_major_formatter(thousands_formatter)

    ax2 = axes_long[1]
    if not df_weekly.empty:
        ax2.plot(df_weekly.index, df_weekly['close'], label='Giá (Tuần)', color='#2ca02c', linewidth=1.5)
        ax2.plot(df_weekly.index, df_weekly['SMA_10_W'], label='SMA 10 Tuần', linestyle='--')
        ax2.plot(df_weekly.index, df_weekly['SMA_40_W'], label='SMA 40 Tuần', linestyle='--')
        buy_signals_w = df_weekly[df_weekly['AI_Signal_W'] == 1]; sell_signals_w = df_weekly[df_weekly['AI_Signal_W'] == -1]
        ax2.scatter(buy_signals_w.index, buy_signals_w['close'] * 0.98, label='AI Mua (Tuần)', marker='^', color='cyan', s=120, zorder=5, edgecolor='black')
        ax2.scatter(sell_signals_w.index, sell_signals_w['close'] * 1.02, label='AI Bán (Tuần)', marker='v', color='magenta', s=120, zorder=5, edgecolor='black')
    ax2.set_title('Phân Tích Khung Tuần và Tín hiệu AI Chiến Lược (Tuần)', fontsize=12)
    ax2.set_ylabel('Giá (VND)'); ax2.legend(); ax2.yaxis.set_major_formatter(thousands_formatter)
    
    axes_long[2].plot(df_to_plot.index, df_to_plot['RSI_14'], color='#9467bd'); axes_long[2].axhline(70, linestyle='--', color='red'); axes_long[2].axhline(30, linestyle='--', color='green')
    axes_long[2].set_title('RSI (Ngày)', fontsize=12)
    
    axes_long[3].plot(df_to_plot.index, df_to_plot['MACD_12_26_9'], label='MACD', color='#2ca02c'); axes_long[3].plot(df_to_plot.index, df_to_plot['MACDs_12_26_9'], label='Signal', color='#d62728', linestyle='--'); axes_long[3].bar(df_to_plot.index, df_to_plot['MACDh_12_26_9'], color='grey', alpha=0.5)
    axes_long[3].set_title('MACD (Ngày)', fontsize=12); axes_long[3].legend()
    
    axes_long[4].bar(df_to_plot.index, df_to_plot['volume'], color='#8c564b', alpha=0.6); axes_long[4].set_title('Khối lượng Giao dịch (Ngày)', fontsize=12); axes_long[4].yaxis.set_major_formatter(thousands_formatter)
    
    ax6 = axes_long[5]
    normalized_stock = (df_to_plot['close'] / df_to_plot['close'].iloc[0]) * 100
    normalized_index = (df_to_plot['vnindex_close'] / df_to_plot['vnindex_close'].iloc[0]) * 100
    ax6.plot(df_to_plot.index, normalized_stock, label=f'Hiệu suất {symbol.upper()}'); ax6.plot(df_to_plot.index, normalized_index, label='Hiệu suất VN-Index', color='black', linestyle='--')
    ax6.set_title('So Sánh Hiệu Suất Tương Đối với VN-Index', fontsize=12)
    ax6.set_ylabel('Hiệu suất (Bắt đầu=100)'); ax6.legend()
    
    chart1_base64 = fig_to_base64(fig_long)
    plt.close(fig_long)

    # --- CỬA SỔ 2: PHÂN TÍCH NGẮN HẠN ---
    fig_short, ax_short = plt.subplots(figsize=(15, 8), constrained_layout=True)
    fig_short.suptitle(f'CỬA SỔ 2: PHÂN TÍCH NGẮN HẠN & DỰ BÁO - {symbol.upper()}', fontsize=18, fontweight='bold')
    short_term_data = df_to_plot.tail(60)
    ax_short.plot(short_term_data.index, short_term_data['close'], label='Giá (60 ngày)', color='#008B8B', marker='o', markersize=3, linestyle='-')
    ax_short.plot(short_term_data.index, short_term_data['SMA_20'], label='SMA 20 ngày', color='#FF00FF', linestyle=':', linewidth=1)
    buy_signals_short = short_term_data[short_term_data['AI_Signal'] == 1]
    sell_signals_short = short_term_data[short_term_data['AI_Signal'] == -1]
    ax_short.scatter(buy_signals_short.index, buy_signals_short['close'] * 0.99, label='AI Mua (Ngày)', marker='^', color='lime', s=120, zorder=5, edgecolor='black')
    ax_short.scatter(sell_signals_short.index, sell_signals_short['close'] * 1.01, label='AI Bán (Ngày)', marker='v', color='red', s=120, zorder=5, edgecolor='black')
    if forecast_price and forecast_date_str:
        ax_short.scatter([pd.to_datetime(forecast_date_str)], [forecast_price], color='orange', marker='*', s=300, label=f'Dự báo ARIMA: {forecast_price:,.0f}', zorder=10, edgecolor='black')
    ax_short.set_title('Diễn biến 60 ngày gần nhất', fontsize=12)
    ax_short.set_ylabel('Giá (VND)'); ax_short.set_xlabel('Ngày'); ax_short.legend()
    ax_short.yaxis.set_major_formatter(thousands_formatter)
    
    chart2_base64 = fig_to_base64(fig_short)
    plt.close(fig_short)
    
    # TRẢ VỀ KẾT QUẢ
    return {
        "symbol": symbol.upper(),
        "last_close_price": f"{last_close_price:,.0f} VND",
        "last_date": latest_data.index[0].strftime('%Y-%m-%d'),
        "ai_recommendation": ai_recommendation,
        "forecast_price": f"{forecast_price:,.0f} VND" if forecast_price else "N/A",
        "chart1": chart1_base64,
        "chart2": chart2_base64
    }

# ENDPOINT ĐỂ SERVE GIAO DIỆN CHÍNH
@app.route('/')
def serve_index():
    # File index.html phải nằm cùng thư mục với server.py
    return send_from_directory('.', 'index.html')

# ENDPOINT ĐỂ PHÂN TÍCH
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    symbol = data.get('symbol')
    if not symbol:
        return jsonify({"error": "Vui lòng nhập mã cổ phiếu."}), 400

    symbol = symbol.upper()
    current_time = datetime.now().timestamp()
    
    # KIỂM TRA CACHE
    if symbol in cache and current_time - cache[symbol]['timestamp'] < CACHE_TIMEOUT:
        return jsonify(cache[symbol]['data'])

    try:
        result = phan_tich_va_du_bao_ai_server(symbol)
        # LƯU VÀO CACHE
        cache[symbol] = {
            'data': result,
            'timestamp': current_time
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Dòng này chỉ dùng để chạy thử trên máy local, không dùng trên Render
    app.run(debug=False, port=5001)

