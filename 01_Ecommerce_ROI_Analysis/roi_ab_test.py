import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import os

# ================= 1. 自动生成模拟数据集 =================
def generate_mock_data(filename="mock_orders.csv"):
    """生成包含异常值和缺失值的电商订单数据"""
    print("正在生成 100,000 条模拟订单数据...")
    np.random.seed(42)
    n_records = 100000
    
    # 模拟订单金额 (大部分在 50-500 之间，混入几个极大的异常值代表刷单/B端采购)
    amounts = np.random.lognormal(mean=4.5, sigma=0.8, size=n_records)
    amounts[np.random.choice(n_records, 100)] = np.random.uniform(10000, 50000, 100) 
    
    # 模拟渠道 (部分缺失)
    channels = np.random.choice(['App', 'WeChat_Mini', 'Douyin', 'Kuaishou', np.nan], 
                                size=n_records, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    
    df = pd.DataFrame({
        'order_id': [f"ORD{str(i).zfill(6)}" for i in range(n_records)],
        'user_id': np.random.randint(1000, 50000, n_records),
        'order_amount': np.round(amounts, 2),
        'channel': channels
    })
    df.to_csv(filename, index=False)
    print(f"数据生成完毕: {filename}\n")

# ================= 2. 数据清洗与业务统计 =================
def clean_and_analyze(filename="mock_orders.csv"):
    df = pd.read_csv(filename)
    initial_count = len(df)
    
    # 填补缺失值
    df['channel'] = df['channel'].fillna('Unknown')
    
    # 使用 IQR 剔除金额异常值
    Q1 = df['order_amount'].quantile(0.25)
    Q3 = df['order_amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 过滤正常数据
    df_clean = df[(df['order_amount'] >= lower_bound) & (df['order_amount'] <= upper_bound)]
    cleaned_count = len(df_clean)
    
    print("--- 数据清洗结果 ---")
    print(f"原始数据量: {initial_count} 条")
    print(f"清洗后数据量: {cleaned_count} 条")
    print(f"共剔除异常大单及无效数据: {initial_count - cleaned_count} 条\n")
    
    print("--- 各渠道客单价分布 (ROI驱动因子初探) ---")
    channel_stat = df_clean.groupby('channel')['order_amount'].mean().round(2).sort_values(ascending=False)
    print(channel_stat)
    print("\n")

# ================= 3. A/B 测试显著性检验 =================
def run_ab_test():
    """验证页面改版对转化率的提升是否显著"""
    print("--- A/B 测试 (页面点击率 CTR 优化) ---")
    
    # 假设实验数据: 对照组(老页面) vs 测试组(新页面)
    control_views, control_clicks = 12000, 960  # CTR 8.00%
    test_views, test_clicks = 12000, 1060      # CTR 8.83% (相对提升 8.3%)
    
    clicks = np.array([test_clicks, control_clicks])
    views = np.array([test_views, control_views])
    
    # Proportions Z-test (单尾检验，测试组 > 对照组)
    z_stat, p_val = proportions_ztest(clicks, views, alternative='larger')
    
    print(f"老版本 CTR: {control_clicks/control_views:.2%}")
    print(f"新版本 CTR: {test_clicks/test_views:.2%}")
    print(f"统计检验结果 -> Z-Statistic: {z_stat:.4f}, P-Value: {p_val:.4f}")
    
    if p_val < 0.05:
        print("业务结论: P值 < 0.05，拒绝原假设！新版页面点击率提升具备统计学显著性，建议全量上线！")
    else:
        print("业务结论: P值 >= 0.05，无显著差异，提升可能源于随机波动，建议持续观察或优化策略。")

# ================= 执行主函数 =================
if __name__ == "__main__":
    generate_mock_data()
    clean_and_analyze()
    run_ab_test()