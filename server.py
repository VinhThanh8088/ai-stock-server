# -*- coding: utf-8 -*-
import pandas as pd
import pandas_ta as ta
from vnstock import Vnstock
import numpy as np
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import BDay
import warnings
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
import json

# Cấu hình để ẩn các cảnh báo không cần thiết
logging.getLogger("vnstock.common.data.data_explorer").setLevel(logging.WARNING)
warnings.filterwarnings('ignore')

# Khởi tạo Web Server
app = Flask(__name__)
# Cho phép ứng dụng web từ mọi nơi có thể gọi đến server này
CORS(app)

# Dùng để cache (lưu trữ) các mô hình đã huấn luyện để không phải huấn luyện lại mỗi lần gọi
# Điều này giúp tăng tốc độ phản hồi đáng kể
model_cache = {}

def phan_tich_va_du_bao_ai(symbol: str):
    """
    Hàm này được điều chỉnh từ mã gốc của bạn.
    Thay vì hiển thị biểu đồ, nó sẽ trả về dữ liệu dạng JSON
    để ứng dụng web có thể đọc được.
    """
    symbol = symbol.upper()
    print(f"🚀 Bắt đầu phân tích cho mã: {symbol} 🚀")

    # 1. TẢI DỮ LIỆU LỊCH SỬ CỔ PHIẾU VÀ VN-INDEX
    print("[1/5] 📥 Đang tải dữ liệu lịch sử...")
    start_date = '2010-01-01'
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    
    df = Vnstock().stock(symbol=symbol).quote.history(start=start_date, end=end_date)
    if df.empty:
        raise ValueError(f"Không tìm thấy dữ liệu cho mã {symbol}")
    
    df_index = Vnstock().stock(symbol='VNINDEX').quote.history(start=start_date, end=end_date)
    
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df_index['time'] = pd.to_datetime(df_index['time'])
    df_index.set_index('time', inplace=True)

    # 2. TÍNH TOÁN CHỈ BÁO VÀ TÍCH HỢP DỮ LIỆU
    print("[2/5] 🛠️  Đang tính toán chỉ báo và tích hợp dữ liệu...")
    df_index.ta.rsi(length=14, append=True); df_index.ta.sma(length=50, append=True)
    df_index.rename(columns={'close': 'vnindex_close', 'RSI_14': 'vnindex_rsi', 'SMA_50': 'vnindex_sma50'}, inplace=True)
    df = df.join(df_index[['vnindex_close', 'vnindex_rsi', 'vnindex_sma50']])

    df.ta.sma(length=20, append=True); df.ta.sma(length=50, append=True)
    df.ta.rsi(length=14, append=True); df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, append=True)
    df.dropna(inplace=True)
    
    df_for_prediction = df.copy()

    # 3. HUẤN LUYỆN MÔ HÌNH AI (HOẶC LẤY TỪ CACHE)
    if symbol in model_cache:
        print(f"[3/5] 🧠 Lấy mô hình AI Giao Dịch (Ngày) từ cache cho {symbol}...")
        model_xgb_daily = model_cache[symbol]
    else:
        print(f"[3/5] 🧠 Đang huấn luyện AI Giao Dịch (khung ngày) cho {symbol}...")
        future_window = 10
        df['price_change_pct'] = (df['close'].shift(-future_window) - df['close']) / df['close'] * 100
        change_threshold = 5.0
        df['Signal'] = np.where(df['price_change_pct'] > change_threshold, 1, np.where(df['price_change_pct'] < -change_threshold, -1, 0))
        df.dropna(subset=['price_change_pct', 'Signal'], inplace=True)

        features = ['open', 'high', 'low', 'close', 'volume', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'vnindex_close', 'vnindex_rsi', 'vnindex_sma50']
        X = df[features]
        y = df['Signal']
        model_xgb_daily = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False, n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
        model_xgb_daily.fit(X, y.map({-1: 0, 0: 1, 1: 2}))
        model_cache[symbol] = model_xgb_daily # Lưu mô hình vào cache
        print(f"✅ Huấn luyện và lưu vào cache thành công.")

    # 4. DỰ BÁO VÀ TỔNG HỢP KHUYẾN NGHỊ
    print("[4/5] 📣 Đang tổng hợp khuyến nghị...")
    features_pred = ['open', 'high', 'low', 'close', 'volume', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'vnindex_close', 'vnindex_rsi', 'vnindex_sma50']
    latest_data = df_for_prediction[features_pred].iloc[-1:]
    
    prediction_mapped = model_xgb_daily.predict(latest_data)[0]
    ai_recommendation = {0: 'BÁN', 1: 'GIỮ', 2: 'MUA'}[prediction_mapped]
    
    df_for_prediction['AI_Signal'] = model_xgb_daily.predict(df_for_prediction[features_pred]).map({0: -1, 1: 0, 2: 1})

    # Chạy dự báo ARIMA
    forecast_price = None
    try:
        print("🔮 Đang chạy mô hình dự báo giá ngày mai (ARIMA)...")
        model_arima = ARIMA(df_for_prediction['close'].tail(200), order=(5,1,0)) # Dùng 200 điểm dữ liệu cuối cho nhanh
        model_arima_fit = model_arima.fit()
        forecast_price = model_arima_fit.forecast(steps=1).iloc[0]
    except Exception as arima_error:
        print(f"⚠️ Không thể chạy mô hình ARIMA. Lỗi: {arima_error}")

    # 5. CHUẨN BỊ DỮ LIỆU GỬI VỀ
    print("[5/5] 📦 Đang đóng gói dữ liệu kết quả...")
    
    # Chuyển đổi index thành chuỗi string để gửi qua JSON
    df_for_prediction.index = df_for_prediction.index.strftime('%Y-%m-%d')
    
    # Chỉ lấy các cột cần thiết để giảm dung lượng gửi đi
    chart_data = df_for_prediction[[
        'close', 'volume', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD_12_26_9', 
        'MACDs_12_26_9', 'MACDh_12_26_9', 'AI_Signal', 'vnindex_close'
    ]].to_dict(orient='index')

    # Chuyển đổi định dạng dữ liệu cho Chart.js
    labels = list(chart_data.keys())
    final_data = {
        'labels': labels,
        'datasets': {key: [d[key] if d and key in d else None for d in chart_data.values()] for key in chart_data[labels[0]].keys()}
    }

    last_close_price = latest_data['close'].values[0]

    return {
        "symbol": symbol,
        "last_price": last_close_price,
        "recommendation": ai_recommendation,
        "forecast_price": forecast_price,
        "chart_data": final_data,
    }

# Đây là "cổng" API mà ứng dụng web sẽ gọi tới
@app.route('/analyze/<symbol>', methods=['GET'])
def analyze_stock(symbol):
    try:
        result = phan_tich_va_du_bao_ai(symbol)
        # Chuyển đổi các kiểu dữ liệu của numpy thành kiểu của Python để jsonify hoạt động
        result['last_price'] = float(result['last_price'])
        if result['forecast_price'] is not None:
            result['forecast_price'] = float(result['forecast_price'])
        
        return jsonify(result)
    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng: {e}")
        return jsonify({"error": str(e)}), 500

# Chạy server khi tệp này được thực thi
if __name__ == '__main__':
    # port=5000 là cổng mặc định, có thể thay đổi
    app.run(host='0.0.0.0', port=5000)
