# generate_bcpn_data.py
from prometheus_api_client import PrometheusConnect
import pandas as pd
import json
from datetime import datetime, timedelta
import os  # 补充遗漏的os模块导入

# 配置Prometheus地址
PROM_URL = "http://localhost:9090"
# 故障时间窗口（根据实际故障注入时间调整）
START_TIME = datetime.now() - timedelta(minutes=20)
END_TIME = datetime.now()
STEP = "5s"  # 与采样间隔一致

# 定义BCPN四层节点与对应Prometheus指标
BCPN_NODES = [
    {
        "node_id": "host01_ssd_io_util",
        "layer": "H",
        "metric": "node_disk_io_time_seconds_total{device=\"sda\"}",
        "has_inherent_defect": True,
        "node_description": "SSD磁盘IO利用率"
    },
    {
        "node_id": "kernel_sched_latency",
        "layer": "K",
        "metric": "node_schedstat_waiting_seconds_total",
        "has_inherent_defect": False,
        "node_description": "内核调度等待延迟"
    },
    {
        "node_id": "docker_mysql_cpu",
        "layer": "R",
        "metric": 'container_cpu_usage_seconds_total{id="/docker/3fd0456b55187f74ec3c93e54288a9290d05a983a180f5847488e5020153c915"}',
        "has_inherent_defect": False,
        "node_description": "MySQL容器CPU使用率"
    },
    {
        "node_id": "order_service_retries",
        "layer": "A",
        "metric": "app_retries_total",
        "has_inherent_defect": False,
        "node_description": "订单服务重试次数"
    }
]

def main():
    prom = PrometheusConnect(url=PROM_URL, disable_ssl=True)
    os.makedirs("bcpn_test_scenario/nodes", exist_ok=True)
    
    # 1. 生成每个节点的CSV文件
    for node in BCPN_NODES:
        print(f"采集节点：{node['node_id']}")
        # 修复点：step参数通过params字典传递，而非直接传入
        data = prom.get_metric_range_data(
            metric_name=node["metric"],
            start_time=START_TIME,
            end_time=END_TIME,
            params={"step": STEP}  # 核心修复：将step放入params参数
        )
        
        if not data:
            print(f"警告：未采集到节点 {node['node_id']} 的数据")
            continue
            
        # 转换为DataFrame
        df = pd.DataFrame(data[0]["values"], columns=["time", "value"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.to_csv(f"bcpn_test_scenario/nodes/{node['node_id']}.csv", index=False)
    
    # 2. 生成场景元数据文件
    scenario_metadata = {
        "scenario_info": {
            "scenario_id": "ssd_fault_with_retry_loop",
            "scenario_name": "SSD介质坏道叠加应用重试风暴",
            "fault_start_time": START_TIME.isoformat(),
            "fault_end_time": END_TIME.isoformat(),
            "affected_hosts": ["host01"]
        },
        "nodes": BCPN_NODES,
        "logs": []
    }
    
    with open("bcpn_test_scenario/scenario_metadata.json", "w", encoding="utf-8") as f:
        json.dump(scenario_metadata, f, ensure_ascii=False, indent=2)
    
    print("✅ BCPN格式数据生成完成，路径：./bcpn_test_scenario/")

if __name__ == "__main__":
    main()
