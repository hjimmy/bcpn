import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta

# ==============================================
# 数据配置（严格对齐BCPN模型参数）
# ==============================================
OUTPUT_DIR = "bcpn_test_scenario"
NODES_DIR = os.path.join(OUTPUT_DIR, "nodes")
START_TIME = datetime(2026, 4, 20, 10, 0, 0)  # 基准时间
TOTAL_POINTS = 60  # 总采样点（10秒/点 = 10分钟）
FAULT_START_POINT = 20  # 第20个点开始故障（3分20秒）
SAMPLING_INTERVAL = 10  # 10秒采样间隔（与论文一致）

# 节点配置（完全匹配BCPN模型）
NODES_CONFIG = [
    {
        "node_id": "host01_ssd_io_util",
        "layer": "H",
        "metric": "node_disk_io_time_seconds_total{device=\"sda\"}",
        "has_inherent_defect": True,
        "node_description": "SSD磁盘IO利用率（根因：硬件老化）",
        "normal_mean": 15,
        "normal_std": 3,
        "fault_mean": 85,
        "fault_std": 10,
        "lag": 0  # 根因无滞后
    },
    {
        "node_id": "kernel_sched_latency",
        "layer": "K",
        "metric": "node_schedstat_waiting_seconds_total",
        "has_inherent_defect": False,
        "node_description": "内核调度等待延迟",
        "normal_mean": 8,
        "normal_std": 2,
        "fault_mean": 60,
        "fault_std": 15,
        "lag": 2  # 滞后SSD 2个点（20秒）
    },
    {
        "node_id": "docker_mysql_cpu",
        "layer": "R",
        "metric": "container_cpu_usage_seconds_total{container_name=\"/mysql\"}",
        "has_inherent_defect": False,
        "node_description": "MySQL容器CPU使用率",
        "normal_mean": 20,
        "normal_std": 5,
        "fault_mean": 65,
        "fault_std": 12,
        "lag": 4  # 滞后SSD 4个点（40秒）
    },
    {
        "node_id": "order_service_retries",
        "layer": "A",
        "metric": "app_retries_total",
        "has_inherent_defect": False,
        "node_description": "订单服务重试次数（自激环放大器）",
        "normal_mean": 2,
        "normal_std": 1,
        "fault_mean": 150,
        "fault_std": 40,
        "lag": 6  # 滞后SSD 6个点（60秒）
    }
]

# ==============================================
# 生成带自激环特征的时序数据
# ==============================================
def generate_timeseries(node_config):
    """生成符合故障传播规律的时序数据，包含自激环特征"""
    normal_mean = node_config["normal_mean"]
    normal_std = node_config["normal_std"]
    fault_mean = node_config["fault_mean"]
    fault_std = node_config["fault_std"]
    lag = node_config["lag"]
    
    # 1. 生成正常阶段数据（前FAULT_START_POINT个点）
    normal_data = np.random.normal(normal_mean, normal_std, FAULT_START_POINT)
    normal_data = np.maximum(normal_data, 0)  # 确保非负
    
    # 2. 生成故障阶段数据（后TOTAL_POINTS-FAULT_START_POINT个点）
    fault_length = TOTAL_POINTS - FAULT_START_POINT
    fault_data = np.zeros(fault_length)
    
    # 故障上升阶段（线性增长）
    rise_length = 15
    for i in range(rise_length):
        progress = i / rise_length
        mean = normal_mean + (fault_mean - normal_mean) * progress
        std = normal_std + (fault_std - normal_std) * progress
        fault_data[i] = np.random.normal(mean, std)
    
    # 自激环阶段（指数增长+波动，体现反馈放大）
    for i in range(rise_length, fault_length):
        # 自激环导致指标二次飙升
        feedback_factor = 1.0 + (i - rise_length) / 20  # 反馈放大系数
        mean = fault_mean * feedback_factor
        std = fault_std * feedback_factor
        fault_data[i] = np.random.normal(mean, std)
    
    # 3. 应用滞后（故障传播延迟）
    if lag > 0:
        # 前lag个点保持正常
        fault_data[:lag] = np.random.normal(normal_mean, normal_std, lag)
    
    # 4. 合并数据并添加微小噪声（避免常量列）
    full_data = np.concatenate([normal_data, fault_data])
    full_data += np.random.normal(0, 0.1, TOTAL_POINTS)  # 微小噪声
    full_data = np.maximum(full_data, 0)
    full_data = np.round(full_data, 4)
    
    return full_data

# ==============================================
# 生成时间戳
# ==============================================
def generate_timestamps():
    timestamps = []
    for i in range(TOTAL_POINTS):
        ts = START_TIME + timedelta(seconds=i * SAMPLING_INTERVAL)
        timestamps.append(ts.isoformat())
    return timestamps

# ==============================================
# 主函数：生成所有数据
# ==============================================
def main():
    # 创建目录
    os.makedirs(NODES_DIR, exist_ok=True)
    print(f"✅ 创建数据目录：{NODES_DIR}")
    
    # 生成统一时间戳
    timestamps = generate_timestamps()
    
    # 生成每个节点的CSV文件
    nodes_meta = []
    for node_config in NODES_CONFIG:
        node_id = node_config["node_id"]
        print(f"\n生成节点数据：{node_id}")
        
        # 生成时序数据
        values = generate_timeseries(node_config)
        
        # 生成DataFrame
        df = pd.DataFrame({
            "time": timestamps,
            "value": values
        })
        
        # 保存CSV
        csv_path = os.path.join(NODES_DIR, f"{node_id}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  保存到：{csv_path}")
        print(f"  数据统计：均值={np.mean(values):.2f}，标准差={np.std(values):.2f}")
        
        # 保存元数据（去掉生成用的参数）
        node_meta = {k: v for k, v in node_config.items() 
                    if k not in ["normal_mean", "normal_std", "fault_mean", "fault_std", "lag"]}
        node_meta["csv_file"] = f"nodes/{node_id}.csv"
        nodes_meta.append(node_meta)
    
    # 生成场景元数据JSON
    scenario_metadata = {
        "scenario_info": {
            "scenario_id": "standard_ssd_retry_loop",
            "scenario_name": "标准SSD故障+应用重试自激环场景",
            "fault_start_time": timestamps[FAULT_START_POINT],
            "fault_end_time": timestamps[-1],
            "affected_hosts": ["host01"],
            "description": "用于验证BCPN模型的标准测试集，包含完整的跨层故障传播和自激反馈环"
        },
        "nodes": nodes_meta,
        "logs": [
            {
                "timestamp": timestamps[FAULT_START_POINT],
                "level": "WARNING",
                "process": "kernel",
                "message": "ata1.00: status: { DRDY ERR }, error: { UNC }"
            },
            {
                "timestamp": timestamps[FAULT_START_POINT + 6],
                "level": "ERROR",
                "process": "order-service",
                "message": "Database connection timeout, retrying..."
            },
            {
                "timestamp": timestamps[FAULT_START_POINT + 15],
                "level": "CRITICAL",
                "process": "prometheus",
                "message": "SSD IO utilization > 90% for 5 minutes"
            }
        ]
    }
    
    json_path = os.path.join(OUTPUT_DIR, "scenario_metadata.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(scenario_metadata, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("🎉 标准测试数据集生成完成！")
    print("="*60)
    print(f"📁 数据根目录：{OUTPUT_DIR}")
    print(f"📄 元数据文件：{json_path}")
    print(f"📊 节点数量：{len(NODES_CONFIG)}")
    print(f"⏱️  时间范围：{timestamps[0]} 至 {timestamps[-1]}")
    print(f"🔍 采样间隔：{SAMPLING_INTERVAL}秒")
    print(f"⚠️  故障开始：{timestamps[FAULT_START_POINT]}")
    print("\n✅ 数据特点：")
    print("  1. 严格遵循故障传播时间线：SSD(20)→内核(22)→MySQL(24)→应用(26)")
    print("  2. 包含自激环特征：第35个点后指标二次飙升")
    print("  3. 所有数据有真实波动，无常量列")
    print("  4. SSD节点标记为固有缺陷，根因明确")

if __name__ == "__main__":
    main()
