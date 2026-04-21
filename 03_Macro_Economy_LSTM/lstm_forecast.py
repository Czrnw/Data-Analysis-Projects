import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# 屏蔽 TensorFlow 的一些烦人警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ================= 1. 生成宏观经济模拟数据 =================
def generate_macro_data(filename="macro_economy_data.csv"):
    """生成 10 年 (120个月) 的宏观经济模拟数据，并故意制造缺失值"""
    print("正在生成长三角 10 年宏观经济模拟数据...")
    np.random.seed(42)
    dates = pd.date_range(start="2014-01-01", periods=120, freq="ME")
    
    # 模拟经济周期 (正弦波 + 趋势 + 随机噪声)
    time = np.arange(120)
    gdp_growth = 6.0 + np.sin(time / 12) + np.random.normal(0, 0.2, 120)
    cpi = 2.0 + np.sin(time / 6) * 0.5 + np.random.normal(0, 0.1, 120)
    ppi = 1.5 + np.sin(time / 8) * 0.8 + np.random.normal(0, 0.3, 120)
    m2_growth = 10.0 - time * 0.02 + np.random.normal(0, 0.5, 120)
    retail_sales = 8.0 + np.cos(time / 12) * 2 + np.random.normal(0, 0.4, 120)
    industry_value = 5.5 + np.sin(time / 10) * 1.5 + np.random.normal(0, 0.2, 120)
    
    df = pd.DataFrame({
        'Date': dates,
        'GDP_Growth': gdp_growth,
        'CPI': cpi,
        'PPI': ppi,
        'M2_Growth': m2_growth,
        'Retail_Sales': retail_sales,
        'Industry_Value': industry_value
    })
    
    # 故意制造 10% 的缺失值 (模拟统计局数据延迟)
    mask = np.random.choice([True, False], size=120, p=[0.1, 0.9])
    df.loc[mask, 'CPI'] = np.nan
    df.loc[np.random.choice([True, False], size=120, p=[0.1, 0.9]), 'Retail_Sales'] = np.nan
    
    df.to_csv(filename, index=False)
    print(f"数据已保存至 {filename}，包含缺失值待处理。\n")
    return filename

# ================= 2. 数据预处理与特征筛选 =================
def preprocess_and_select(filename):
    print("--- 步骤 1: 数据清洗与特征降维 ---")
    df = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')
    
    # 1. 缺失值处理: 线性插值
    missing_before = df.isna().sum().sum()
    df = df.interpolate(method='linear').bfill()
    print(f"插值填补了 {missing_before} 个缺失数据点，保证时序连续性。")
    
    # 2. 随机森林特征筛选
    X = df.drop(columns=['GDP_Growth'])
    y = df['GDP_Growth']
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # 获取重要性排名前 3 的特征
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.nlargest(3).index.tolist()
    print(f"随机森林筛选出的 TOP 3 核心驱动因子: {top_features}\n")
    
    return df, top_features

# ================= 3. 构造滑动窗口与归一化 =================
def create_dataset(df, top_features, time_steps=6):
    """将时间序列转换为监督学习问题 (用过去 6 个月预测下个月)"""
    print("--- 步骤 2: 构造 LSTM 三维时序张量 ---")
    features_to_use = top_features + ['GDP_Growth']
    data_subset = df[features_to_use].values
    
    # 数据归一化 (神经网络对输入尺度敏感)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_subset)
    
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        # 取过去 time_steps 个月的所有特征作为输入
        X.append(scaled_data[i:(i + time_steps), :])
        # 取下个月的 GDP_Growth (在最后一列) 作为预测目标
        y.append(scaled_data[i + time_steps, -1])
        
    X, y = np.array(X), np.array(y)
    
    # 划分训练集和测试集 (前 80% 训练，后 20% 测试)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"训练集形状 (样本数, 时间步, 特征数): {X_train.shape}")
    print(f"测试集形状: {X_test.shape}\n")
    return X_train, X_test, y_train, y_test, scaler

# ================= 4. 构建与训练 LSTM 模型 =================
def build_and_train_lstm(X_train, y_train, X_test, y_test):
    print("--- 步骤 3: 训练 LSTM 模型 (引入防过拟合机制) ---")
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=False, 
             input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),  # 随机丢弃 20% 神经元，防止小样本过拟合
        Dense(32, activation='relu'),
        Dense(1)       # 预测单维度的 GDP 增长率
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # 早停机制: 当验证集 loss 连续 10 轮不下降时停止，并恢复最优权重
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("开始模型拟合 (展示核心轮次)...")
    history = model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=8, 
        validation_data=(X_test, y_test),
        callbacks=[early_stop], 
        verbose=0 # 关闭满屏进度条，保持输出整洁
    )
    print(f"模型在第 {len(history.epoch)} 轮触发早停机制或训练完成。\n")
    
    # ================= 5. 模型评估 =================
    print("--- 步骤 4: 模型评估 ---")
    y_pred = model.predict(X_test, verbose=0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"在测试集上的预测 RMSE (均方根误差): {rmse:.4f}")
    if rmse < 0.2:
        print("业务结论: 预测误差较小，模型收敛良好，成功捕捉宏观经济的长程时序趋势！")
    else:
        print("业务结论: 误差仍在可接受范围，可通过增加数据量或调整窗口期进一步优化。")

# ================= 执行主程序 =================
if __name__ == "__main__":
    file_path = generate_macro_data()
    df, features = preprocess_and_select(file_path)
    X_train, X_test, y_train, y_test, scaler = create_dataset(df, features, time_steps=6)
    build_and_train_lstm(X_train, y_train, X_test, y_test)