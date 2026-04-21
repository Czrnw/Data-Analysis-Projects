import pandas as pd
import os

# ================= 1. 生成两份带异常的模拟数据 =================
def generate_audit_data():
    """生成 ERP 系统数据与第三方支付平台流水数据"""
    
    # ERP 系统的应收账款记录
    erp_data = {
        'order_id': ['ORD001', 'ORD002', 'ORD003', 'ORD004', 'ORD005', 'ORD006'],
        'expected_amount': [100.50, 200.00, 350.80, 45.00, 88.00, 500.00],
        'order_date': ['2025-01-01'] * 6
    }
    
    # 支付网关的实际流水记录 (故意制造异常)
    pay_data = {
        'transaction_id': ['TXN101', 'TXN102', 'TXN103', 'TXN104', 'TXN105', 'TXN106', 'TXN107'],
        'order_id': ['ORD001', 'ORD002', 'ORD003', 'ORD003', 'ORD004', 'ORD005', 'ORD099'],
        # ORD003 被分两次支付: 300.00 + 50.80 = 350.80 (正常)
        # ORD005 支付金额少了: 80.00 != 88.00 (异常: 少付)
        # ORD006 支付网关没有记录 (异常: 漏单)
        # ORD099 ERP系统没有记录 (异常: 幽灵流水)
        'actual_paid': [100.50, 200.00, 300.00, 50.80, 45.00, 80.00, 150.00]
    }
    
    pd.DataFrame(erp_data).to_csv("erp_orders.csv", index=False)
    pd.DataFrame(pay_data).to_csv("payment_gateway.csv", index=False)
    print("模拟对账文件 (erp_orders.csv, payment_gateway.csv) 已生成。\n")

# ================= 2. 自动化对账引擎 =================
def run_automated_audit(erp_file, pay_file, report_file="audit_exception_report.csv"):
    print("--- 开始执行自动化对账脚本 ---")
    
    df_erp = pd.read_csv(erp_file)
    df_pay = pd.read_csv(pay_file)
    
    # 步骤 1: 格式清洗 (去除两端不可见空格，防止 Join 失败)
    df_erp['order_id'] = df_erp['order_id'].astype(str).str.strip()
    df_pay['order_id'] = df_pay['order_id'].astype(str).str.strip()
    
    # 步骤 2: 预聚合第三方流水 (解决 1对N 支付问题)
    # 将属于同一个 order_id 的多笔流水金额累加
    df_pay_agg = df_pay.groupby('order_id', as_index=False).agg(
        total_paid=('actual_paid', 'sum'),
        transaction_count=('transaction_id', 'count') # 记录拆单次数
    )
    
    # 步骤 3: 多源异构数据全外连接 (Outer Join)
    df_merge = pd.merge(df_erp, df_pay_agg, on='order_id', how='outer')
    
    # 步骤 4: 填充空值 (未匹配上的用 0 替代)
    df_merge['expected_amount'] = df_merge['expected_amount'].fillna(0)
    df_merge['total_paid'] = df_merge['total_paid'].fillna(0)
    
    # 步骤 5: 计算差异 (处理 Python 浮点数精度问题，四舍五入保留2位小数)
    df_merge['diff_amount'] = round(df_merge['total_paid'] - df_merge['expected_amount'], 2)
    
    # 步骤 6: 设定容错阈值，筛选异常记录
    tolerance = 0.01
    df_anomalies = df_merge[df_merge['diff_amount'].abs() > tolerance].copy()
    
    # 标记异常原因
    def flag_issue(row):
        if row['expected_amount'] == 0: return "ERP系统漏单 (幽灵支付)"
        if row['total_paid'] == 0: return "支付网关漏单 (用户未付款)"
        if row['diff_amount'] > 0: return "实付金额大于应付"
        if row['diff_amount'] < 0: return "实付金额小于应付"
        return "未知错误"
        
    df_anomalies['issue_type'] = df_anomalies.apply(flag_issue, axis=1)
    
    # 导出并展示结果
    df_anomalies.to_csv(report_file, index=False)
    print(f"对账成功！发现 {len(df_anomalies)} 笔金额异常订单。")
    print("\n--- 异常订单明细 (audit_exception_report.csv) ---")
    print(df_anomalies[['order_id', 'expected_amount', 'total_paid', 'diff_amount', 'issue_type']].to_string(index=False))

# ================= 3. 执行入口 =================
if __name__ == "__main__":
    generate_audit_data()
    run_automated_audit('erp_orders.csv', 'payment_gateway.csv')