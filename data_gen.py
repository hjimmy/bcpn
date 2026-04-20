import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict

# ==============================================
# 全局配置
# ==============================================
BASE_OUTPUT_DIR = "os_fault_dataset"
TOTAL_POINTS = 60  # 每个场景60个采样点（10秒/点 = 10分钟）
FAULT_START_POINT = 20  # 第20个点开始故障
SAMPLING_INTERVAL = 10  # 10秒采样间隔
START_TIME = datetime(2026, 4, 20, 10, 0, 0)

# ==============================================
# 32个故障场景完整定义
# ==============================================
FAULT_SCENARIOS = [
    # ==================== 硬件层(H) 8个故障 ====================
    {
        "scenario_id": "H01_ssd_io_high",
        "scenario_name": "SSD磁盘IO利用率飙升（介质坏道）",
        "root_cause_layer": "H",
        "nodes": [
            {
                "node_id": "host01_ssd_io_util",
                "layer": "H",
                "has_inherent_defect": True,
                "node_description": "SSD磁盘IO利用率（根因）",
                "normal_mean": 15, "normal_std": 3,
                "fault_mean": 90, "fault_std": 8,
                "lag": 0
            },
            {
                "node_id": "kernel_io_wait",
                "layer": "K",
                "has_inherent_defect": False,
                "node_description": "内核IO等待时间",
                "normal_mean": 5, "normal_std": 2,
                "fault_mean": 80, "fault_std": 20,
                "lag": 2
            },
            {
                "node_id": "docker_app_cpu",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "应用容器CPU使用率",
                "normal_mean": 25, "normal_std": 5,
                "fault_mean": 75, "fault_std": 15,
                "lag": 4
            },
            {
                "node_id": "app_io_errors",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用IO错误次数",
                "normal_mean": 1, "normal_std": 0.5,
                "fault_mean": 100, "fault_std": 30,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "H02_cpu_overheat",
        "scenario_name": "CPU温度过高（散热故障）",
        "root_cause_layer": "H",
        "nodes": [
            {
                "node_id": "host01_cpu_temp",
                "layer": "H",
                "has_inherent_defect": True,
                "node_description": "CPU温度（根因：散热风扇故障）",
                "normal_mean": 45, "normal_std": 3,
                "fault_mean": 92, "fault_std": 5,
                "lag": 0
            },
            {
                "node_id": "kernel_cpu_throttle",
                "layer": "K",
                "has_inherent_defect": False,
                "node_description": "CPU降频计数",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 50, "fault_std": 10,
                "lag": 2
            },
            {
                "node_id": "docker_app_latency",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器响应延迟",
                "normal_mean": 50, "normal_std": 10,
                "fault_mean": 300, "fault_std": 80,
                "lag": 4
            },
            {
                "node_id": "app_timeouts",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用超时次数",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 80, "fault_std": 20,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "H03_memory_ecc_error",
        "scenario_name": "内存ECC错误（硬件故障）",
        "root_cause_layer": "H",
        "nodes": [
            {
                "node_id": "host01_ecc_errors",
                "layer": "H",
                "has_inherent_defect": True,
                "node_description": "内存ECC错误计数（根因）",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 200, "fault_std": 50,
                "lag": 0
            },
            {
                "node_id": "kernel_page_faults",
                "layer": "K",
                "has_inherent_defect": False,
                "node_description": "内核页错误计数",
                "normal_mean": 100, "normal_std": 20,
                "fault_mean": 5000, "fault_std": 1000,
                "lag": 2
            },
            {
                "node_id": "docker_app_restarts",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器重启次数",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 10, "fault_std": 3,
                "lag": 4
            },
            {
                "node_id": "app_crashes",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用崩溃次数",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 25, "fault_std": 8,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "H04_network_packet_loss",
        "scenario_name": "网络接口丢包（网卡故障）",
        "root_cause_layer": "H",
        "nodes": [
            {
                "node_id": "host01_net_drop",
                "layer": "H",
                "has_inherent_defect": True,
                "node_description": "网络接口丢包率（根因）",
                "normal_mean": 0.1, "normal_std": 0.05,
                "fault_mean": 15, "fault_std": 5,
                "lag": 0
            },
            {
                "node_id": "kernel_net_retrans",
                "layer": "K",
                "has_inherent_defect": False,
                "node_description": "TCP重传率",
                "normal_mean": 0.5, "normal_std": 0.2,
                "fault_mean": 25, "fault_std": 8,
                "lag": 2
            },
            {
                "node_id": "docker_net_latency",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器网络延迟",
                "normal_mean": 20, "normal_std": 5,
                "fault_mean": 500, "fault_std": 150,
                "lag": 4
            },
            {
                "node_id": "app_net_timeouts",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用网络超时",
                "normal_mean": 1, "normal_std": 0.5,
                "fault_mean": 120, "fault_std": 40,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "H05_disk_full",
        "scenario_name": "磁盘空间满（inode耗尽）",
        "root_cause_layer": "H",
        "nodes": [
            {
                "node_id": "host01_disk_usage",
                "layer": "H",
                "has_inherent_defect": True,
                "node_description": "磁盘空间使用率（根因）",
                "normal_mean": 60, "normal_std": 5,
                "fault_mean": 99.5, "fault_std": 0.3,
                "lag": 0
            },
            {
                "node_id": "kernel_fs_errors",
                "layer": "K",
                "has_inherent_defect": False,
                "node_description": "文件系统错误计数",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 50, "fault_std": 15,
                "lag": 2
            },
            {
                "node_id": "docker_log_errors",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器日志写入错误",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 30, "fault_std": 10,
                "lag": 4
            },
            {
                "node_id": "app_write_failures",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用写入失败次数",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 90, "fault_std": 25,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "H06_raid_degraded",
        "scenario_name": "RAID阵列降级（硬盘掉线）",
        "root_cause_layer": "H",
        "nodes": [
            {
                "node_id": "host01_raid_status",
                "layer": "H",
                "has_inherent_defect": True,
                "node_description": "RAID阵列状态（根因：1=degraded）",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 1, "fault_std": 0,
                "lag": 0
            },
            {
                "node_id": "kernel_raid_sync",
                "layer": "K",
                "has_inherent_defect": False,
                "node_description": "RAID同步IO延迟",
                "normal_mean": 10, "normal_std": 3,
                "fault_mean": 200, "fault_std": 50,
                "lag": 2
            },
            {
                "node_id": "docker_disk_latency",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器磁盘延迟",
                "normal_mean": 5, "normal_std": 2,
                "fault_mean": 150, "fault_std": 40,
                "lag": 4
            },
            {
                "node_id": "app_disk_slow",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用磁盘操作慢查询",
                "normal_mean": 2, "normal_std": 1,
                "fault_mean": 60, "fault_std": 20,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "H07_power_supply",
        "scenario_name": "电源供应故障（冗余电源失效）",
        "root_cause_layer": "H",
        "nodes": [
            {
                "node_id": "host01_power_status",
                "layer": "H",
                "has_inherent_defect": True,
                "node_description": "电源状态（根因：1=故障）",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 1, "fault_std": 0,
                "lag": 0
            },
            {
                "node_id": "kernel_voltage_fluct",
                "layer": "K",
                "has_inherent_defect": False,
                "node_description": "电压波动计数",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 100, "fault_std": 30,
                "lag": 2
            },
            {
                "node_id": "docker_unexpected_exit",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器异常退出次数",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 15, "fault_std": 5,
                "lag": 4
            },
            {
                "node_id": "app_data_corruption",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用数据校验错误",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 40, "fault_std": 12,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "H08_pcie_error",
        "scenario_name": "PCIe总线错误",
        "root_cause_layer": "H",
        "nodes": [
            {
                "node_id": "host01_pcie_aer",
                "layer": "H",
                "has_inherent_defect": True,
                "node_description": "PCIe AER错误计数（根因）",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 300, "fault_std": 80,
                "lag": 0
            },
            {
                "node_id": "kernel_pci_reset",
                "layer": "K",
                "has_inherent_defect": False,
                "node_description": "PCI设备重置计数",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 20, "fault_std": 6,
                "lag": 2
            },
            {
                "node_id": "docker_device_error",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器设备访问错误",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 50, "fault_std": 15,
                "lag": 4
            },
            {
                "node_id": "app_device_fail",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用设备操作失败",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 70, "fault_std": 22,
                "lag": 6
            }
        ]
    },
    
    # ==================== 内核层(K) 8个故障 ====================
    {
        "scenario_id": "K01_sched_latency",
        "scenario_name": "内核调度延迟过高",
        "root_cause_layer": "K",
        "nodes": [
            {
                "node_id": "kernel_sched_latency",
                "layer": "K",
                "has_inherent_defect": True,
                "node_description": "内核调度延迟（根因）",
                "normal_mean": 8, "normal_std": 2,
                "fault_mean": 120, "fault_std": 30,
                "lag": 0
            },
            {
                "node_id": "docker_cpu_wait",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器CPU等待时间",
                "normal_mean": 5, "normal_std": 2,
                "fault_mean": 80, "fault_std": 20,
                "lag": 2
            },
            {
                "node_id": "app_response_slow",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用响应时间",
                "normal_mean": 30, "normal_std": 8,
                "fault_mean": 400, "fault_std": 100,
                "lag": 4
            },
            {
                "node_id": "app_retries",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用重试次数",
                "normal_mean": 1, "normal_std": 0.5,
                "fault_mean": 90, "fault_std": 25,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "K02_syscall_latency",
        "scenario_name": "系统调用延迟增加",
        "root_cause_layer": "K",
        "nodes": [
            {
                "node_id": "kernel_syscall_latency",
                "layer": "K",
                "has_inherent_defect": True,
                "node_description": "系统调用平均延迟（根因）",
                "normal_mean": 0.1, "normal_std": 0.03,
                "fault_mean": 5, "fault_std": 1.5,
                "lag": 0
            },
            {
                "node_id": "docker_syscall_count",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器系统调用计数",
                "normal_mean": 10000, "normal_std": 2000,
                "fault_mean": 50000, "fault_std": 10000,
                "lag": 2
            },
            {
                "node_id": "app_func_latency",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用函数调用延迟",
                "normal_mean": 1, "normal_std": 0.3,
                "fault_mean": 20, "fault_std": 5,
                "lag": 4
            },
            {
                "node_id": "app_timeouts",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用超时次数",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 60, "fault_std": 18,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "K03_oom_killer",
        "scenario_name": "OOM内存回收频繁",
        "root_cause_layer": "K",
        "nodes": [
            {
                "node_id": "kernel_oom_score",
                "layer": "K",
                "has_inherent_defect": True,
                "node_description": "OOM Killer触发次数（根因）",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 10, "fault_std": 3,
                "lag": 0
            },
            {
                "node_id": "docker_mem_usage",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器内存使用率",
                "normal_mean": 60, "normal_std": 10,
                "fault_mean": 95, "fault_std": 3,
                "lag": 2
            },
            {
                "node_id": "docker_oom_events",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器OOM事件",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 8, "fault_std": 2,
                "lag": 4
            },
            {
                "node_id": "app_restarts",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用重启次数",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 15, "fault_std": 4,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "K04_net_buffer_overflow",
        "scenario_name": "网络栈缓冲区溢出",
        "root_cause_layer": "K",
        "nodes": [
            {
                "node_id": "kernel_net_buf_overflow",
                "layer": "K",
                "has_inherent_defect": True,
                "node_description": "网络缓冲区溢出计数（根因）",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 200, "fault_std": 50,
                "lag": 0
            },
            {
                "node_id": "kernel_net_drop",
                "layer": "K",
                "has_inherent_defect": False,
                "node_description": "内核网络丢包",
                "normal_mean": 1, "normal_std": 0.5,
                "fault_mean": 50, "fault_std": 15,
                "lag": 2
            },
            {
                "node_id": "docker_net_queue",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器网络队列长度",
                "normal_mean": 10, "normal_std": 3,
                "fault_mean": 200, "fault_std": 50,
                "lag": 4
            },
            {
                "node_id": "app_net_errors",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用网络错误",
                "normal_mean": 1, "normal_std": 0.5,
                "fault_mean": 80, "fault_std": 22,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "K05_fs_journal_lag",
        "scenario_name": "文件系统日志同步延迟",
        "root_cause_layer": "K",
        "nodes": [
            {
                "node_id": "kernel_journal_lag",
                "layer": "K",
                "has_inherent_defect": True,
                "node_description": "文件系统日志同步延迟（根因）",
                "normal_mean": 0.01, "normal_std": 0.005,
                "fault_mean": 2, "fault_std": 0.5,
                "lag": 0
            },
            {
                "node_id": "kernel_fs_sync",
                "layer": "K",
                "has_inherent_defect": False,
                "node_description": "fsync调用延迟",
                "normal_mean": 0.1, "normal_std": 0.03,
                "fault_mean": 3, "fault_std": 0.8,
                "lag": 2
            },
            {
                "node_id": "docker_disk_sync",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器磁盘同步延迟",
                "normal_mean": 0.5, "normal_std": 0.2,
                "fault_mean": 5, "fault_std": 1.5,
                "lag": 4
            },
            {
                "node_id": "app_transaction_fail",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用事务失败",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 40, "fault_std": 12,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "K06_softirq_high",
        "scenario_name": "内核软中断过高",
        "root_cause_layer": "K",
        "nodes": [
            {
                "node_id": "kernel_softirq",
                "layer": "K",
                "has_inherent_defect": True,
                "node_description": "软中断CPU占比（根因）",
                "normal_mean": 5, "normal_std": 2,
                "fault_mean": 45, "fault_std": 10,
                "lag": 0
            },
            {
                "node_id": "kernel_hardirq",
                "layer": "K",
                "has_inherent_defect": False,
                "node_description": "硬中断CPU占比",
                "normal_mean": 3, "normal_std": 1,
                "fault_mean": 20, "fault_std": 5,
                "lag": 2
            },
            {
                "node_id": "docker_cpu_steal",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器CPU被抢占时间",
                "normal_mean": 1, "normal_std": 0.5,
                "fault_mean": 30, "fault_std": 8,
                "lag": 4
            },
            {
                "node_id": "app_cpu_starve",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用CPU饥饿",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 50, "fault_std": 15,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "K07_context_switch",
        "scenario_name": "进程上下文切换频繁",
        "root_cause_layer": "K",
        "nodes": [
            {
                "node_id": "kernel_ctxt",
                "layer": "K",
                "has_inherent_defect": True,
                "node_description": "上下文切换次数/秒（根因）",
                "normal_mean": 5000, "normal_std": 1000,
                "fault_mean": 150000, "fault_std": 30000,
                "lag": 0
            },
            {
                "node_id": "kernel_runq",
                "layer": "K",
                "has_inherent_defect": False,
                "node_description": "运行队列长度",
                "normal_mean": 2, "normal_std": 1,
                "fault_mean": 50, "fault_std": 15,
                "lag": 2
            },
            {
                "node_id": "docker_threads",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器线程数",
                "normal_mean": 50, "normal_std": 10,
                "fault_mean": 500, "fault_std": 100,
                "lag": 4
            },
            {
                "node_id": "app_thread_spawn",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用线程创建失败",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 30, "fault_std": 10,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "K08_kernel_deadlock",
        "scenario_name": "内核死锁/活锁",
        "root_cause_layer": "K",
        "nodes": [
            {
                "node_id": "kernel_lock_contention",
                "layer": "K",
                "has_inherent_defect": True,
                "node_description": "内核锁竞争计数（根因）",
                "normal_mean": 10, "normal_std": 3,
                "fault_mean": 500, "fault_std": 100,
                "lag": 0
            },
            {
                "node_id": "kernel_load_avg",
                "layer": "K",
                "has_inherent_defect": False,
                "node_description": "系统load average",
                "normal_mean": 2, "normal_std": 0.5,
                "fault_mean": 50, "fault_std": 10,
                "lag": 2
            },
            {
                "node_id": "docker_process_block",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器阻塞进程数",
                "normal_mean": 1, "normal_std": 0.5,
                "fault_mean": 30, "fault_std": 8,
                "lag": 4
            },
            {
                "node_id": "app_unresponsive",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用无响应",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 1, "fault_std": 0,
                "lag": 6
            }
        ]
    },
    
    # ==================== 运行时层(R) 8个故障 ====================
    {
        "scenario_id": "R01_cpu_throttle",
        "scenario_name": "Docker容器CPU配额耗尽",
        "root_cause_layer": "R",
        "nodes": [
            {
                "node_id": "docker_cpu_throttle",
                "layer": "R",
                "has_inherent_defect": True,
                "node_description": "CPU throttling时间（根因）",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 60, "fault_std": 15,
                "lag": 0
            },
            {
                "node_id": "docker_cpu_usage",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器CPU使用率",
                "normal_mean": 40, "normal_std": 10,
                "fault_mean": 95, "fault_std": 3,
                "lag": 2
            },
            {
                "node_id": "app_response_time",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用响应时间",
                "normal_mean": 40, "normal_std": 10,
                "fault_mean": 350, "fault_std": 80,
                "lag": 4
            },
            {
                "node_id": "app_retries",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用重试次数",
                "normal_mean": 1, "normal_std": 0.5,
                "fault_mean": 70, "fault_std": 20,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "R02_container_oom",
        "scenario_name": "容器内存OOM被Kill",
        "root_cause_layer": "R",
        "nodes": [
            {
                "node_id": "docker_oom_killed",
                "layer": "R",
                "has_inherent_defect": True,
                "node_description": "容器OOM被Kill次数（根因）",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 5, "fault_std": 2,
                "lag": 0
            },
            {
                "node_id": "docker_mem_usage",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器内存使用率",
                "normal_mean": 50, "normal_std": 10,
                "fault_mean": 98, "fault_std": 1,
                "lag": 2
            },
            {
                "node_id": "docker_restarts",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器重启次数",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 8, "fault_std": 2,
                "lag": 4
            },
            {
                "node_id": "app_downtime",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用不可用时间",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 40, "fault_std": 10,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "R03_disk_io_throttle",
        "scenario_name": "容器磁盘IO限制",
        "root_cause_layer": "R",
        "nodes": [
            {
                "node_id": "docker_io_throttled",
                "layer": "R",
                "has_inherent_defect": True,
                "node_description": "IO throttled字节数（根因）",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 100000000, "fault_std": 20000000,
                "lag": 0
            },
            {
                "node_id": "docker_io_latency",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器IO延迟",
                "normal_mean": 5, "normal_std": 2,
                "fault_mean": 200, "fault_std": 50,
                "lag": 2
            },
            {
                "node_id": "app_io_wait",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用IO等待时间",
                "normal_mean": 10, "normal_std": 3,
                "fault_mean": 300, "fault_std": 80,
                "lag": 4
            },
            {
                "node_id": "app_io_timeouts",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用IO超时",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 50, "fault_std": 15,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "R04_port_exhaustion",
        "scenario_name": "容器网络端口耗尽",
        "root_cause_layer": "R",
        "nodes": [
            {
                "node_id": "docker_port_usage",
                "layer": "R",
                "has_inherent_defect": True,
                "node_description": "端口使用率（根因）",
                "normal_mean": 20, "normal_std": 5,
                "fault_mean": 98, "fault_std": 1,
                "lag": 0
            },
            {
                "node_id": "docker_conn_wait",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "连接等待数",
                "normal_mean": 5, "normal_std": 2,
                "fault_mean": 200, "fault_std": 50,
                "lag": 2
            },
            {
                "node_id": "app_conn_fail",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用连接失败",
                "normal_mean": 1, "normal_std": 0.5,
                "fault_mean": 80, "fault_std": 20,
                "lag": 4
            },
            {
                "node_id": "app_conn_retries",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用连接重试",
                "normal_mean": 1, "normal_std": 0.5,
                "fault_mean": 120, "fault_std": 30,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "R05_fd_exhaustion",
        "scenario_name": "容器文件描述符耗尽",
        "root_cause_layer": "R",
        "nodes": [
            {
                "node_id": "docker_fd_usage",
                "layer": "R",
                "has_inherent_defect": True,
                "node_description": "文件描述符使用率（根因）",
                "normal_mean": 30, "normal_std": 8,
                "fault_mean": 99, "fault_std": 0.5,
                "lag": 0
            },
            {
                "node_id": "docker_fd_errors",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "FD错误计数",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 100, "fault_std": 25,
                "lag": 2
            },
            {
                "node_id": "app_fd_fail",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用FD打开失败",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 60, "fault_std": 15,
                "lag": 4
            },
            {
                "node_id": "app_crashes",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用崩溃",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 20, "fault_std": 5,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "R06_nproc_limit",
        "scenario_name": "容器进程数达到上限",
        "root_cause_layer": "R",
        "nodes": [
            {
                "node_id": "docker_nproc_usage",
                "layer": "R",
                "has_inherent_defect": True,
                "node_description": "进程数使用率（根因）",
                "normal_mean": 40, "normal_std": 10,
                "fault_mean": 99, "fault_std": 0.5,
                "lag": 0
            },
            {
                "node_id": "docker_fork_fail",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "fork失败计数",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 80, "fault_std": 20,
                "lag": 2
            },
            {
                "node_id": "app_process_spawn",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用进程创建失败",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 50, "fault_std": 12,
                "lag": 4
            },
            {
                "node_id": "app_service_unavail",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "服务不可用",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 1, "fault_std": 0,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "R07_image_pull_fail",
        "scenario_name": "容器镜像拉取失败",
        "root_cause_layer": "R",
        "nodes": [
            {
                "node_id": "docker_image_pull_err",
                "layer": "R",
                "has_inherent_defect": True,
                "node_description": "镜像拉取失败次数（根因）",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 20, "fault_std": 5,
                "lag": 0
            },
            {
                "node_id": "docker_pending_containers",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "Pending容器数",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 15, "fault_std": 4,
                "lag": 2
            },
            {
                "node_id": "app_scale_fail",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用扩容失败",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 10, "fault_std": 3,
                "lag": 4
            },
            {
                "node_id": "app_capacity_insufficient",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "容量不足",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 1, "fault_std": 0,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "R08_health_check_fail",
        "scenario_name": "容器健康检查失败",
        "root_cause_layer": "R",
        "nodes": [
            {
                "node_id": "docker_health_fail",
                "layer": "R",
                "has_inherent_defect": True,
                "node_description": "健康检查连续失败（根因）",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 5, "fault_std": 1,
                "lag": 0
            },
            {
                "node_id": "docker_container_restart",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器重启次数",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 8, "fault_std": 2,
                "lag": 2
            },
            {
                "node_id": "app_flapping",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用频繁重启",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 1, "fault_std": 0,
                "lag": 4
            },
            {
                "node_id": "app_service_unavail",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "服务不可用",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 1, "fault_std": 0,
                "lag": 6
            }
        ]
    },
    
    # ==================== 应用层(A) 8个故障（操作系统相关） ====================
    {
        "scenario_id": "A01_file_lock_contention",
        "scenario_name": "系统级文件锁竞争",
        "root_cause_layer": "A",
        "nodes": [
            {
                "node_id": "app_file_lock_wait",
                "layer": "A",
                "has_inherent_defect": True,
                "node_description": "文件锁等待时间（根因）",
                "normal_mean": 0.1, "normal_std": 0.05,
                "fault_mean": 5, "fault_std": 1.5,
                "lag": 0
            },
            {
                "node_id": "app_lock_contention",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "锁竞争计数",
                "normal_mean": 5, "normal_std": 2,
                "fault_mean": 200, "fault_std": 50,
                "lag": 2
            },
            {
                "node_id": "kernel_lock_wait",
                "layer": "K",
                "has_inherent_defect": False,
                "node_description": "内核锁等待",
                "normal_mean": 1, "normal_std": 0.5,
                "fault_mean": 50, "fault_std": 15,
                "lag": 4
            },
            {
                "node_id": "docker_cpu_iowait",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器IO wait",
                "normal_mean": 5, "normal_std": 2,
                "fault_mean": 40, "fault_std": 10,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "A02_shared_mem_error",
        "scenario_name": "共享内存段错误",
        "root_cause_layer": "A",
        "nodes": [
            {
                "node_id": "app_shm_errors",
                "layer": "A",
                "has_inherent_defect": True,
                "node_description": "共享内存错误计数（根因）",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 50, "fault_std": 15,
                "lag": 0
            },
            {
                "node_id": "app_segfault",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "段错误次数",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 20, "fault_std": 5,
                "lag": 2
            },
            {
                "node_id": "kernel_shm_cleanup",
                "layer": "K",
                "has_inherent_defect": False,
                "node_description": "共享内存清理计数",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 30, "fault_std": 8,
                "lag": 4
            },
            {
                "node_id": "docker_crashes",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器崩溃次数",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 15, "fault_std": 4,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "A03_semaphore_deadlock",
        "scenario_name": "信号量死锁",
        "root_cause_layer": "A",
        "nodes": [
            {
                "node_id": "app_sem_wait",
                "layer": "A",
                "has_inherent_defect": True,
                "node_description": "信号量等待时间（根因）",
                "normal_mean": 0.01, "normal_std": 0.005,
                "fault_mean": 10, "fault_std": 3,
                "lag": 0
            },
            {
                "node_id": "app_process_block",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用阻塞进程数",
                "normal_mean": 1, "normal_std": 0.5,
                "fault_mean": 50, "fault_std": 15,
                "lag": 2
            },
            {
                "node_id": "kernel_sem_ops",
                "layer": "K",
                "has_inherent_defect": False,
                "node_description": "内核信号量操作",
                "normal_mean": 100, "normal_std": 20,
                "fault_mean": 1000, "fault_std": 200,
                "lag": 4
            },
            {
                "node_id": "app_unresponsive",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用无响应",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 1, "fault_std": 0,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "A04_syscall_timeout",
        "scenario_name": "系统调用超时",
        "root_cause_layer": "A",
        "nodes": [
            {
                "node_id": "app_syscall_timeout",
                "layer": "A",
                "has_inherent_defect": True,
                "node_description": "系统调用超时次数（根因）",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 40, "fault_std": 10,
                "lag": 0
            },
            {
                "node_id": "app_retry_syscall",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "系统调用重试",
                "normal_mean": 1, "normal_std": 0.5,
                "fault_mean": 100, "fault_std": 25,
                "lag": 2
            },
            {
                "node_id": "kernel_syscall_load",
                "layer": "K",
                "has_inherent_defect": False,
                "node_description": "内核系统调用负载",
                "normal_mean": 5000, "normal_std": 1000,
                "fault_mean": 50000, "fault_std": 10000,
                "lag": 4
            },
            {
                "node_id": "docker_syscall_latency",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器系统调用延迟",
                "normal_mean": 0.1, "normal_std": 0.03,
                "fault_mean": 2, "fault_std": 0.5,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "A05_env_var_missing",
        "scenario_name": "环境变量配置错误",
        "root_cause_layer": "A",
        "nodes": [
            {
                "node_id": "app_env_errors",
                "layer": "A",
                "has_inherent_defect": True,
                "node_description": "环境变量错误计数（根因）",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 30, "fault_std": 8,
                "lag": 0
            },
            {
                "node_id": "app_config_fail",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "配置加载失败",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 15, "fault_std": 4,
                "lag": 2
            },
            {
                "node_id": "app_start_fail",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用启动失败",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 10, "fault_std": 3,
                "lag": 4
            },
            {
                "node_id": "docker_container_crash",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器崩溃",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 8, "fault_std": 2,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "A06_dynamic_lib_load_fail",
        "scenario_name": "动态库加载失败",
        "root_cause_layer": "A",
        "nodes": [
            {
                "node_id": "app_dlopen_fail",
                "layer": "A",
                "has_inherent_defect": True,
                "node_description": "动态库加载失败（根因）",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 25, "fault_std": 6,
                "lag": 0
            },
            {
                "node_id": "app_symbol_error",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "符号解析错误",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 40, "fault_std": 10,
                "lag": 2
            },
            {
                "node_id": "app_crash",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用崩溃",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 15, "fault_std": 4,
                "lag": 4
            },
            {
                "node_id": "docker_restart_loop",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器重启循环",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 1, "fault_std": 0,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "A07_syslog_block",
        "scenario_name": "系统日志写入阻塞",
        "root_cause_layer": "A",
        "nodes": [
            {
                "node_id": "app_syslog_latency",
                "layer": "A",
                "has_inherent_defect": True,
                "node_description": "系统日志写入延迟（根因）",
                "normal_mean": 0.05, "normal_std": 0.02,
                "fault_mean": 3, "fault_std": 0.8,
                "lag": 0
            },
            {
                "node_id": "app_log_queue",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "日志队列长度",
                "normal_mean": 10, "normal_std": 3,
                "fault_mean": 500, "fault_std": 100,
                "lag": 2
            },
            {
                "node_id": "kernel_syslog_load",
                "layer": "K",
                "has_inherent_defect": False,
                "node_description": "内核syslog负载",
                "normal_mean": 100, "normal_std": 20,
                "fault_mean": 2000, "fault_std": 400,
                "lag": 4
            },
            {
                "node_id": "docker_log_driver_error",
                "layer": "R",
                "has_inherent_defect": False,
                "node_description": "容器日志驱动错误",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 20, "fault_std": 5,
                "lag": 6
            }
        ]
    },
    {
        "scenario_id": "A08_priority_inversion",
        "scenario_name": "进程优先级反转",
        "root_cause_layer": "A",
        "nodes": [
            {
                "node_id": "app_prio_inversion",
                "layer": "A",
                "has_inherent_defect": True,
                "node_description": "优先级反转计数（根因）",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 100, "fault_std": 25,
                "lag": 0
            },
            {
                "node_id": "app_high_prio_wait",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "高优先级进程等待",
                "normal_mean": 0, "normal_std": 0,
                "fault_mean": 50, "fault_std": 12,
                "lag": 2
            },
            {
                "node_id": "kernel_sched_migrate",
                "layer": "K",
                "has_inherent_defect": False,
                "node_description": "进程迁移计数",
                "normal_mean": 100, "normal_std": 20,
                "fault_mean": 2000, "fault_std": 400,
                "lag": 4
            },
            {
                "node_id": "app_response_slow",
                "layer": "A",
                "has_inherent_defect": False,
                "node_description": "应用响应慢",
                "normal_mean": 30, "normal_std": 8,
                "fault_mean": 300, "fault_std": 70,
                "lag": 6
            }
        ]
    }
]

# ==============================================
# 数据生成函数
# ==============================================
def generate_timeseries(node_config):
    """生成符合故障传播规律的时序数据"""
    normal_mean = node_config["normal_mean"]
    normal_std = node_config["normal_std"]
    fault_mean = node_config["fault_mean"]
    fault_std = node_config["fault_std"]
    lag = node_config["lag"]
    
    # 1. 正常阶段
    normal_data = np.random.normal(normal_mean, normal_std, FAULT_START_POINT)
    normal_data = np.maximum(normal_data, 0)
    
    # 2. 故障阶段
    fault_length = TOTAL_POINTS - FAULT_START_POINT
    fault_data = np.zeros(fault_length)
    
    # 故障上升阶段
    rise_length = 15
    for i in range(rise_length):
        progress = i / rise_length
        mean = normal_mean + (fault_mean - normal_mean) * progress
        std = normal_std + (fault_std - normal_std) * progress
        fault_data[i] = np.random.normal(mean, std)
    
    # 自激环阶段（指数增长）
    for i in range(rise_length, fault_length):
        feedback_factor = 1.0 + (i - rise_length) / 20
        mean = fault_mean * feedback_factor
        std = fault_std * feedback_factor
        fault_data[i] = np.random.normal(mean, std)
    
    # 3. 应用滞后
    if lag > 0:
        fault_data[:lag] = np.random.normal(normal_mean, normal_std, lag)
    
    # 4. 合并并添加噪声
    full_data = np.concatenate([normal_data, fault_data])
    full_data += np.random.normal(0, 0.1, TOTAL_POINTS)
    full_data = np.maximum(full_data, 0)
    full_data = np.round(full_data, 4)
    
    return full_data

def generate_timestamps():
    """生成统一时间戳"""
    timestamps = []
    for i in range(TOTAL_POINTS):
        ts = START_TIME + timedelta(seconds=i * SAMPLING_INTERVAL)
        timestamps.append(ts.isoformat())
    return timestamps

def generate_scenario(scenario_config):
    """生成单个故障场景的所有数据"""
    scenario_id = scenario_config["scenario_id"]
    scenario_dir = os.path.join(BASE_OUTPUT_DIR, scenario_id)
    nodes_dir = os.path.join(scenario_dir, "nodes")
    os.makedirs(nodes_dir, exist_ok=True)
    
    timestamps = generate_timestamps()
    nodes_meta = []
    
    for node_config in scenario_config["nodes"]:
        node_id = node_config["node_id"]
        values = generate_timeseries(node_config)
        
        # 保存CSV
        df = pd.DataFrame({"time": timestamps, "value": values})
        csv_path = os.path.join(nodes_dir, f"{node_id}.csv")
        df.to_csv(csv_path, index=False)
        
        # 保存元数据
        node_meta = {k: v for k, v in node_config.items() 
                    if k not in ["normal_mean", "normal_std", "fault_mean", "fault_std", "lag"]}
        node_meta["csv_file"] = f"nodes/{node_id}.csv"
        nodes_meta.append(node_meta)
    
    # 生成场景元数据JSON
    scenario_metadata = {
        "scenario_info": {
            "scenario_id": scenario_config["scenario_id"],
            "scenario_name": scenario_config["scenario_name"],
            "root_cause_layer": scenario_config["root_cause_layer"],
            "fault_start_time": timestamps[FAULT_START_POINT],
            "fault_end_time": timestamps[-1],
            "affected_hosts": ["host01"]
        },
        "nodes": nodes_meta,
        "logs": [
            {
                "timestamp": timestamps[FAULT_START_POINT],
                "level": "WARNING",
                "process": "system",
                "message": f"{scenario_config['scenario_name']} 开始"
            }
        ]
    }
    
    json_path = os.path.join(scenario_dir, "scenario_metadata.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(scenario_metadata, f, ensure_ascii=False, indent=2)
    
    return scenario_dir

# ==============================================
# 主函数
# ==============================================
def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    print(f"✅ 创建根目录：{BASE_OUTPUT_DIR}")
    print(f"📊 总故障场景数：{len(FAULT_SCENARIOS)}")
    print(f"⏱️  每个场景采样点数：{TOTAL_POINTS}")
    print(f"⚠️  故障开始点：{FAULT_START_POINT}")
    print("\n" + "="*60)
    
    success_count = 0
    for i, scenario in enumerate(FAULT_SCENARIOS):
        print(f"\n[{i+1}/{len(FAULT_SCENARIOS)}] 生成场景：{scenario['scenario_name']}")
        try:
            scenario_dir = generate_scenario(scenario)
            print(f"  ✅ 保存到：{scenario_dir}")
            success_count += 1
        except Exception as e:
            print(f"  ❌ 生成失败：{str(e)}")
    
    print("\n" + "="*60)
    print(f"\n🎉 数据生成完成！")
    print(f"✅ 成功生成：{success_count}/{len(FAULT_SCENARIOS)} 个场景")
    print(f"📁 数据根目录：{BASE_OUTPUT_DIR}")
    print("\n📋 场景分类统计：")
    layer_counts = {}
    for s in FAULT_SCENARIOS:
        layer = s["root_cause_layer"]
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    for layer, count in sorted(layer_counts.items()):
        layer_name = {"H": "硬件层", "K": "内核层", "R": "运行时层", "A": "应用层"}[layer]
        print(f"  {layer_name}({layer})：{count} 个")

if __name__ == "__main__":
    main()
