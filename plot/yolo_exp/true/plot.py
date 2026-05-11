import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置图表中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义文件路径
files = [
    "YOLOv5n.csv",
    "YOLOv8n.csv", 
    "YOLOv9t.csv",
    "YOLOv10n.csv",
    "YOLOv11n.csv",
    "YOLOv12n.csv"
]

# 定义颜色列表，用于不同模型的线条
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

# 创建数据读取函数
def read_yolo_csv(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 去除所有列名的前导和尾随空格
    df.columns = df.columns.str.strip()
    
    # 根据不同模型的列名格式，提取所需数据
    if 'metrics/precision' in df.columns:
        # YOLOv5n和YOLOv9t的列名格式
        epoch = df['epoch']
        precision = df['metrics/precision']
        recall = df['metrics/recall']
        mAP50 = df['metrics/mAP_0.5']
        mAP50_95 = df['metrics/mAP_0.5:0.95']
    elif 'metrics/precision(B)' in df.columns:
        # YOLOv8n、v10n、v11n和v12n的列名格式
        epoch = df['epoch']
        precision = df['metrics/precision(B)']
        recall = df['metrics/recall(B)']
        mAP50 = df['metrics/mAP50(B)']
        mAP50_95 = df['metrics/mAP50-95(B)']
    else:
        # 如果列名格式不符合预期，抛出异常
        raise ValueError(f"Unknown column format in file {file_path}")
    
    return epoch, precision, recall, mAP50, mAP50_95

# 创建绘图函数
def plot_metric(metric_data, metric_name, save_path=None):
    plt.figure(figsize=(8, 6))
    
    for i, (model_name, data) in enumerate(metric_data.items()):
        plt.plot(data['epoch'], data[metric_name], label=model_name, color=colors[i],linewidth=2)
    
    plt.title(f'{metric_name}变化曲线', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel(metric_name, fontsize=20)
    # 坐标值的字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, alpha=0.3)
    # 图例字体大小
    plt.legend(fontsize=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

# 主函数
def main():
    # 读取所有文件的数据
    data = {}
    for file in files:
        file_path = os.path.join(r'E:\OUC_wangle\plot\yolo_exp\true', file)
        model_name = os.path.splitext(file)[0]
        try:
            epoch, precision, recall, mAP50, mAP50_95 = read_yolo_csv(file_path)
            data[model_name] = {
                'epoch': epoch,
                'precision': precision,
                'recall': recall,
                'mAP_0.5': mAP50,
                'mAP_0.5:0.95': mAP50_95
            }
            print(f"成功读取 {model_name} 的数据")
        except Exception as e:
            print(f"读取 {model_name} 时出错: {e}")
    
    # 绘制四张图表
    metrics = ['precision', 'recall', 'mAP_0.5', 'mAP_0.5:0.95']
    for metric in metrics:
        # 将文件名中的冒号替换为短横线，避免Windows系统不支持的字符
        safe_metric_name = metric.replace(':', '-')
        prefix = "E:/OUC_wangle/plot/yolo_exp/true"
        save_path = os.path.join(prefix, f"{safe_metric_name}_curve.png")
        # save_path = f"{safe_metric_name}_curve.png"
        plot_metric(data, metric, save_path)
        print(f"已保存 {metric} 曲线图: {save_path}")

if __name__ == "__main__":
    main()