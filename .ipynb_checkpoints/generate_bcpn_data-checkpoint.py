from prometheus_api_client import PrometheusConnect
import pandas as pd
import json
import os
from datetime import datetime, timedelta

# ===================== 配置 =====================
PROM_URL = "http://localhost:9090"
START_TIME = datetime.now() - timedelta(minutes=100)
END_TIME = datetime.now()

# BCPN 四层节点（真实Linux环境指标）
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
        "node_id": "docker_cpu_usage",
        "layer": "R",
        "metric": "container_cpu_usage_seconds_total",
        "has_inherent_defect": False,
        "node_description": "容器CPU使用率"
    },
    {
        "node_id": "app_retries_count",
        "layer": "A",
        "metric": "app_retries_total",
        "has_inherent_defect": False,
        "node_description": "应用重试次数"
    }
]

def main():
    prom = PrometheusConnect(url=PROM_URL, disable_ssl=True)
    os.makedirs("bcpn_test_scenario", exist_ok=True)

    for node in BCPN_NODES:
        print(f"正在采集：{node['node_id']}")

        try:
            # 【修复】去掉不兼容的 step 参数，兼容所有版本
            data = prom.get_metric_range_data(
                metric_name=node["metric"],
                start_time=START_TIME,
                end_time=END_TIME
            )
        except Exception as e:
            print(f"⚠️  采集失败: {e}，跳过该指标")
            continue

        if not data:
            print(f"⚠️  无数据，跳过")
            continue

        # 解析时间与值
        try:
            values = data[0]["values"]
            df = pd.DataFrame(values, columns=["time", "value"])
            df["time"] = pd.to_datetime(df["time"], unit="s")
            csv_path = f"bcpn_test_scenario/{node['node_id']}.csv"
            df.to_csv(csv_path, index=False)
            print(f"✅ 保存成功：{csv_path}")
        except Exception as e:
            print(f"❌ 解析失败: {e}")

    # 生成 BCPN 元数据文件
    metadata = {
        "scenario_info": {
            "scenario_id": "linux_real_fault",
            "name": "真实Linux环境故障模拟",
            "time_start": START_TIME.isoformat(),
            "time_end": END_TIME.isoformat()
        },
        "nodes": [
            {**n, "csv_file": f"{n['node_id']}.csv"} for n in BCPN_NODES
        ],
        "logs": []
    }

    with open("bcpn_test_scenario/scenario_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n🎉 全部完成！数据目录：bcpn_test_scenario/")

if __name__ == "__main__":
    main()
