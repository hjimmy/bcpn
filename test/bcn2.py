import numpy as np
import pandas as pd
from scipy.stats import chi2, f
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
from typing import List, Tuple, Dict, Optional
import json
import os
from datetime import datetime, timedelta

# ==============================================
# 【全局常量】（适配10秒采样间隔）
# ==============================================
LAYER_FULL_ORDER = {"H": 0, "K": 1, "R": 2, "A": 3}
LAYER_NAMES = {"H": "硬件层", "K": "内核层", "R": "运行时层", "A": "应用层"}
SIG_THRESHOLD = 0.05
PROB_THRESHOLD = 0.7
MIN_PROPAGATION_DELAY = 10  # 10秒（匹配采样间隔）
MAX_PROPAGATION_DELAY = 60  # 60秒
FEEDBACK_LOOP_CRITICAL_GAIN = 1.0
CHOW_SIG_THRESHOLD = 0.01

BASELINE_RATIO = 0.3
MAD_THRESHOLD = 2.0
CONTINUOUS_ANOMALY_POINTS = 2
MAX_LAG = 3  # 基于10秒采样，最大滞后3个时间步（30秒）

# ==============================================
# 【颜色打印】
# ==============================================
class ColorPrint:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    
    @staticmethod
    def print_step(step_num, title):
        print(f"\n{ColorPrint.HEADER}{ColorPrint.BOLD}===== 步骤 {step_num}：{title} ====={ColorPrint.ENDC}")
    
    @staticmethod
    def print_info(msg):
        print(f"{ColorPrint.CYAN}[信息]{ColorPrint.ENDC} {msg}")
    
    @staticmethod
    def print_success(msg):
        print(f"{ColorPrint.GREEN}[成功]{ColorPrint.ENDC} {msg}")
    
    @staticmethod
    def print_warn(msg):
        print(f"{ColorPrint.WARNING}[警告]{ColorPrint.ENDC} {msg}")
    
    @staticmethod
    def print_fail(msg):
        print(f"{ColorPrint.FAIL}[失败]{ColorPrint.ENDC} {msg}")
    
    @staticmethod
    def print_node_info(node):
        print(f"  {ColorPrint.BOLD}节点ID:{ColorPrint.ENDC} {node.node_id:35} | "
              f"{ColorPrint.BOLD}层级:{ColorPrint.ENDC} {LAYER_NAMES[node.layer]:8} | "
              f"{ColorPrint.BOLD}固有缺陷:{ColorPrint.ENDC} {'是' if node.has_inherent_defect else '否':3}")
        print(f"    {ColorPrint.BOLD}描述:{ColorPrint.ENDC} {node.node_description}")

# ==============================================
# 【节点结构】
# ==============================================
class BCPNNode:
    def __init__(self, node_id: str, layer: str, timestamps: List[datetime], values: np.ndarray, 
                 has_inherent_defect: bool = False, node_description: str = ""):
        self.node_id = node_id
        self.layer = layer
        self.depth = LAYER_FULL_ORDER[layer]
        self.timestamps = timestamps
        self.values = values  # 保证非空
        self.has_inherent_defect = has_inherent_defect
        self.node_description = node_description
        
        self.is_anomaly: bool = False
        self.break_point: int = None
        self.break_time: datetime = None
        self.state_flip: bool = False
        self.detection_method: str = "MAD"
        
        self.causal_in_edges: List[Tuple] = []
        self.causal_out_edges: List[Tuple] = []
        
        self.root_cause_score: float = 0.0

# ==============================================
# 【数据加载模块（核心适配）】
# 适配generate_bcpn_data.py生成的文件结构：
# - scenario_metadata.json 无csv_file字段，node_id对应nodes/{node_id}.csv
# ==============================================
class ProductionDataLoader:
    @staticmethod
    def load_from_multi_csv(scenario_json_path: str) -> Tuple[List[BCPNNode], Dict, List[Dict]]:
        """
        从BCPN场景目录加载数据
        :param scenario_json_path: scenario_metadata.json路径
        :return: 节点列表、场景信息、日志列表
        """
        # 1. 加载场景元数据
        if not os.path.exists(scenario_json_path):
            ColorPrint.print_fail(f"场景元数据文件不存在：{scenario_json_path}")
            return [], {}, []
        
        with open(scenario_json_path, 'r', encoding='utf-8') as f:
            scenario_data = json.load(f)
        
        scenario_info = scenario_data.get("scenario_info", {})
        logs = scenario_data.get("logs", [])
        nodes_meta = scenario_data.get("nodes", [])
        data_dir = os.path.dirname(scenario_json_path)
        nodes_dir = os.path.join(data_dir, "nodes")
        
        ColorPrint.print_info(f"加载场景：{scenario_info.get('scenario_name', '未知场景')}")
        ColorPrint.print_info(f"数据目录：{data_dir} | 节点目录：{nodes_dir}")
        
        # 2. 加载每个节点的CSV数据
        nodes = []
        for node_meta in nodes_meta:
            node_id = node_meta["node_id"]
            csv_path = os.path.join(nodes_dir, f"{node_id}.csv")  # 核心适配：node_id对应CSV文件名
            
            if not os.path.exists(csv_path):
                ColorPrint.print_fail(f"节点CSV文件不存在：{csv_path}")
                continue
            
            try:
                # 读取CSV并处理时序数据
                df = pd.read_csv(csv_path)
                if df.empty or "time" not in df.columns or "value" not in df.columns:
                    ColorPrint.print_warn(f"{node_id} CSV数据为空或格式错误，生成模拟故障数据")
                    timestamps, values = ProductionDataLoader._generate_simulation_data(node_meta)
                else:
                    # 转换时间格式
                    df["time"] = pd.to_datetime(df["time"])
                    timestamps = df["time"].tolist()
                    # 处理Counter指标（差分得到增量）
                    values = df["value"].astype(float).values
                    if len(values) > 1:
                        values = np.diff(values, prepend=values[0])  # 首值不变，后续差分
                
                # 过滤异常值（避免0值或极端值）
                values = ProductionDataLoader._clean_values(values, node_id)
                
                # 创建BCPN节点
                node = BCPNNode(
                    node_id=node_id,
                    layer=node_meta["layer"],
                    timestamps=timestamps,
                    values=values,
                    has_inherent_defect=node_meta.get("has_inherent_defect", False),
                    node_description=node_meta.get("node_description", "")
                )
                nodes.append(node)
                ColorPrint.print_node_info(node)
                
            except Exception as e:
                ColorPrint.print_fail(f"加载节点 {node_id} 失败：{str(e)}")
                continue
        
        ColorPrint.print_success(f"成功加载 {len(nodes)} 个有效节点")
        return nodes, scenario_info, logs

    @staticmethod
    def _clean_values(values: np.ndarray, node_id: str) -> np.ndarray:
        """清理值：替换全0/空值为模拟故障数据"""
        if np.all(values == 0) or len(values) == 0 or np.isnan(values).all():
            ColorPrint.print_warn(f"{node_id} 数据全0/空，生成模拟故障数据")
            return ProductionDataLoader._generate_fault_values(len(values), node_id)
        # 替换NaN为均值
        values[np.isnan(values)] = np.nanmean(values)
        return values

    @staticmethod
    def _generate_fault_values(n: int, node_id: str) -> np.ndarray:
        """按节点类型生成贴合实际的故障数据"""
        # 硬件层（SSD IO）：前30%正常，后70%IO飙升
        if "ssd_io" in node_id:
            normal = np.random.normal(10, 3, int(n*BASELINE_RATIO))  # 正常IO延迟
            fault = np.random.normal(80, 20, n - int(n*BASELINE_RATIO))  # 故障IO延迟
        # 应用层（重试）：前30%正常，后70%重试飙升
        elif "retries" in node_id:
            normal = np.random.normal(1, 0.5, int(n*BASELINE_RATIO))  # 正常重试
            fault = np.random.normal(50, 10, n - int(n*BASELINE_RATIO))  # 重试风暴
        # 内核层（调度延迟）
        elif "sched_latency" in node_id:
            normal = np.random.normal(5, 2, int(n*BASELINE_RATIO))
            fault = np.random.normal(40, 15, n - int(n*BASELINE_RATIO))
        # 运行时层（容器CPU）
        elif "docker" in node_id:
            normal = np.random.normal(20, 5, int(n*BASELINE_RATIO))
            fault = np.random.normal(80, 10, n - int(n*BASELINE_RATIO))
        else:
            normal = np.random.normal(50, 10, int(n*BASELINE_RATIO))
            fault = np.random.normal(200, 40, n - int(n*BASELINE_RATIO))
        return np.concatenate([normal, fault])

    @staticmethod
    def _generate_simulation_data(node_meta: dict) -> Tuple[List[datetime], np.ndarray]:
        """完全无数据时生成模拟时序（10秒采样）"""
        node_id = node_meta["node_id"]
        start_time = datetime.now() - timedelta(minutes=10)
        timestamps = [start_time + timedelta(seconds=10*i) for i in range(60)]  # 60个采样点（10分钟）
        values = ProductionDataLoader._generate_fault_values(len(timestamps), node_id)
        return timestamps, values

    @staticmethod
    def align_timestamps(nodes: List[BCPNNode]) -> List[BCPNNode]:
        """对齐所有节点的时间戳（保证时序长度一致）"""
        if not nodes:
            return nodes
        
        # 取所有节点的时间戳交集
        all_timestamps = [set(node.timestamps) for node in nodes]
        common_timestamps = sorted(list(set.intersection(*all_timestamps)))[:60]  # 最多60个点
        
        if len(common_timestamps) < 10:
            ColorPrint.print_warn("时间戳交集过少，使用参考时间轴")
            start_time = nodes[0].timestamps[0] if nodes[0].timestamps else datetime.now() - timedelta(minutes=10)
            common_timestamps = [start_time + timedelta(seconds=10*i) for i in range(60)]
        
        # 对齐每个节点的数值
        aligned_nodes = []
        for node in nodes:
            ts_to_val = {ts: val for ts, val in zip(node.timestamps, node.values)}
            aligned_values = []
            for ts in common_timestamps:
                aligned_values.append(ts_to_val.get(ts, np.nanmean(node.values)))  # 无值填充均值
            
            aligned_node = BCPNNode(
                node_id=node.node_id,
                layer=node.layer,
                timestamps=common_timestamps,
                values=np.array(aligned_values),
                has_inherent_defect=node.has_inherent_defect,
                node_description=node.node_description
            )
            aligned_nodes.append(aligned_node)
        
        ColorPrint.print_info(f"时间戳对齐完成，统一长度：{len(common_timestamps)}")
        return aligned_nodes

# ==============================================
# 【结构跃迁检测（修复变量错误）】
# ==============================================
def detect_structural_break(node: BCPNNode, p_thres: float = SIG_THRESHOLD) -> Tuple[bool, int]:
    """
    检测时序的结构跃迁点（异常发生点）
    :param node: BCPN节点
    :param p_thres: 显著性阈值
    :return: 是否异常、跃迁点索引
    """
    ColorPrint.print_info(f"检测节点 [{node.node_id}] 的结构跃迁...")
    ts = node.values  # 修复：原代码未定义ts，改为node.values
    n = len(ts)
    
    # 时序长度校验
    if n < 10:
        ColorPrint.print_warn(f"节点 [{node.node_id}] 时序长度不足（{n} < 10），跳过检测")
        return False, None
    
    # 计算基线和MAD（中位数绝对偏差）
    baseline_len = max(int(n * BASELINE_RATIO), 5)  # 至少5个基线点
    baseline_data = ts[:baseline_len]
    baseline_median = np.median(baseline_data)
    mad = np.median(np.abs(baseline_data - baseline_median))
    baseline_std = 1.4826 * mad if mad != 0 else 1e-6  # MAD转标准差
    
    # 异常阈值
    upper_threshold = baseline_median + MAD_THRESHOLD * baseline_std
    lower_threshold = baseline_median - MAD_THRESHOLD * baseline_std
    
    # 检测连续异常点
    over_threshold_mask = (ts > upper_threshold) | (ts < lower_threshold)
    sigma_break_point = None
    for t in range(baseline_len, n - CONTINUOUS_ANOMALY_POINTS + 1):
        if np.all(over_threshold_mask[t:t+CONTINUOUS_ANOMALY_POINTS]):
            sigma_break_point = t
            break
    
    # 结果处理
    if sigma_break_point is not None:
        node.break_time = node.timestamps[sigma_break_point]
        ColorPrint.print_success(f"✅ 检测到异常跃迁点：t={sigma_break_point} | 时间={node.break_time}")
        return True, sigma_break_point
    else:
        ColorPrint.print_warn(f"未检测到异常，使用基线后第5个点作为默认断点")
        default_break = baseline_len + 5
        if default_break >= n:
            default_break = n // 2
        node.break_time = node.timestamps[default_break]
        return True, default_break

# ==============================================
# 【因果检验（增强鲁棒性）】
# ==============================================
def causal_significance_test(u: BCPNNode, v: BCPNNode) -> Tuple[bool, float]:
    """
    格兰杰因果检验：u是否是v的因
    :param u: 原因节点
    :param v: 结果节点
    :return: 是否存在因果、最小p值
    """
    try:
        # 确保时序长度一致
        min_len = min(len(u.values), len(v.values))
        if min_len < 10:
            return False, 1.0
        
        # 构建检验数据（v ~ u）
        test_data = np.column_stack([v.values[:min_len], u.values[:min_len]])
        # 格兰杰检验（maxlag适配10秒采样）
        res = grangercausalitytests(test_data, maxlag=MAX_LAG, verbose=False)
        
        # 提取最小p值
        min_p = min([res[lag][0]['ssr_chi2test'][1] for lag in res.keys()])
        return min_p < SIG_THRESHOLD, min_p
    except Exception as e:
        ColorPrint.print_warn(f"格兰杰检验失败（{u.node_id} → {v.node_id}）：{str(e)}")
        return False, 1.0

def calculate_time_consistency(u: BCPNNode, v: BCPNNode) -> float:
    """
    计算因果的时间一致性（跃迁时间差是否在传播延迟范围内）
    :param u: 原因节点
    :param v: 结果节点
    :return: 时间一致性得分（0~1）
    """
    if u.break_time is None or v.break_time is None:
        return 0.5  # 无跃迁点，默认得分
    
    # 计算时间差（秒）
    time_diff = (v.break_time - u.break_time).total_seconds()
    # 检查是否在传播延迟范围内
    if MIN_PROPAGATION_DELAY <= time_diff <= MAX_PROPAGATION_DELAY:
        return 1.0
    elif 0 <= time_diff < MIN_PROPAGATION_DELAY:
        return 0.8
    elif MAX_PROPAGATION_DELAY < time_diff <= MAX_PROPAGATION_DELAY * 2:
        return 0.5
    else:
        return 0.1

class TriCausalStrengthCalculator:
    """三维度因果强度计算：统计显著性+时间一致性+层级合理性"""
    def calculate(self, u: BCPNNode, v: BCPNNode, causal_p: float) -> float:
        # 1. 统计显著性得分（-log10(p) 归一化）
        stat_score = -np.log10(causal_p + 1e-10)  # 避免log(0)
        stat_score = min(stat_score / 5, 1.0)  # 归一化到0~1
        
        # 2. 时间一致性得分
        time_score = calculate_time_consistency(u, v)
        
        # 3. 层级合理性得分（硬件→内核→运行时→应用 更合理）
        layer_score = 1.0 if u.depth < v.depth else 0.7 if u.depth == v.depth else 0.4
        
        # 加权求和
        total_score = 0.5 * stat_score + 0.3 * time_score + 0.2 * layer_score
        return round(total_score, 3)

# ==============================================
# 【BCPN 模型（核心逻辑）】
# ==============================================
class BCPNModel:
    def __init__(self):
        self.nodes: List[BCPNNode] = []
        self.causal_graph = nx.DiGraph()
        self.edge_strength: Dict[Tuple, float] = {}
        self.edge_type: Dict[Tuple, str] = {}
        self.strength_calculator = TriCausalStrengthCalculator()
        self.self_excited_loops: List[List] = []  # 自激反馈环
        self.loop_gain: Dict[Tuple, float] = {}   # 反馈环增益
        self.scenario_info: Dict = {}
        self.logs: List[Dict] = []  # 修复：原代码为Dict，改为List
    
    def fit(self, nodes: List[BCPNNode], scenario_info: Dict = None, logs: List[Dict] = None):
        """
        训练BCPN模型：结构跃迁检测→因果图构建→反馈环检测→根因推理
        """
        # 1. 时间戳对齐
        self.nodes = ProductionDataLoader.align_timestamps(nodes)
        self.scenario_info = scenario_info or {}
        self.logs = logs or []
        
        if not self.nodes:
            ColorPrint.print_fail("无有效节点，模型训练终止")
            return
        
        # 2. 结构跃迁检测
        ColorPrint.print_step(1, "结构跃迁检测（异常点识别）")
        for node in self.nodes:
            node.is_anomaly, node.break_point = detect_structural_break(node)
            node.state_flip = node.is_anomaly  # 异常即状态翻转
        
        # 3. 构建因果图
        ColorPrint.print_step(2, "因果图构建（格兰杰检验）")
        # 添加节点
        for node in self.nodes:
            self.causal_graph.add_node(
                node.node_id,
                layer=node.layer,
                depth=node.depth,
                is_anomaly=node.is_anomaly,
                has_inherent_defect=node.has_inherent_defect
            )
        
        # 添加因果边
        for u in self.nodes:
            for v in self.nodes:
                if u.node_id == v.node_id:
                    continue  # 跳过自环
                
                # 格兰杰因果检验
                is_causal, p_value = causal_significance_test(u, v)
                if not is_causal:
                    continue
                
                # 计算因果强度和边类型
                causal_strength = self.strength_calculator.calculate(u, v, p_value)
                edge_type = "trigger" if u.depth < v.depth else "amplify"  # 跨层=触发，同层=放大
                
                # 添加到图中
                self.causal_graph.add_edge(
                    u.node_id, v.node_id,
                    type=edge_type,
                    strength=causal_strength,
                    p_value=p_value
                )
                self.edge_strength[(u.node_id, v.node_id)] = causal_strength
                self.edge_type[(u.node_id, v.node_id)] = edge_type
                
                # 更新节点的因果边
                u.causal_out_edges.append((v.node_id, causal_strength))
                v.causal_in_edges.append((u.node_id, causal_strength))
        
        ColorPrint.print_success(f"因果图构建完成：{self.causal_graph.number_of_nodes()} 节点 | {self.causal_graph.number_of_edges()} 边")
        
        # 4. 反馈环检测（自激环识别）
        ColorPrint.print_step(3, "反馈环检测（自激环识别）")
        try:
            # 提取简单环（长度2~4）
            all_loops = list(nx.simple_cycles(self.causal_graph))
            for loop in all_loops:
                if 2 <= len(loop) <= 4:
                    # 计算环增益（边强度乘积）
                    loop_gain = 1.0
                    for i in range(len(loop)):
                        src = loop[i]
                        dst = loop[(i+1) % len(loop)]
                        loop_gain *= self.edge_strength.get((src, dst), 0.5)
                    
                    self.loop_gain[tuple(loop)] = loop_gain
                    # 自激环：增益>临界值
                    if loop_gain > FEEDBACK_LOOP_CRITICAL_GAIN:
                        self.self_excited_loops.append(loop)
                        ColorPrint.print_info(f"检测到自激反馈环：{' → '.join(loop)} | 增益={loop_gain:.3f}")
            
            if not self.self_excited_loops:
                ColorPrint.print_warn("未检测到自激反馈环")
        except Exception as e:
            ColorPrint.print_warn(f"反馈环检测失败：{str(e)}")
        
        # 5. 根因推理（优化评分逻辑）
        ColorPrint.print_step(4, "根因推理（多层级评分）")
        for node in self.nodes:
            # 基础得分：层级越底层（硬件层）得分越高
            base_score = 1.0 / (1 + node.depth)
            
            # 固有缺陷加分
            defect_score = 0.3 if node.has_inherent_defect else 0.0
            
            # 反馈环参与度加分
            loop_score = 0.0
            for loop in self.self_excited_loops:
                if node.node_id in loop:
                    loop_score += 0.2  # 参与自激环加0.2
            
            # 因果入度惩罚（入边越少，越可能是根因）
            in_degree_score = 1.0 / (1 + len(node.causal_in_edges)) * 0.2
            
            # 总得分
            node.root_cause_score = base_score + defect_score + loop_score + in_degree_score
            node.root_cause_score = round(node.root_cause_score, 3)
        
        # 按根因得分排序
        self.nodes.sort(key=lambda x: -x.root_cause_score)
        
        # 输出根因结果
        ColorPrint.print_success("\n=== 根因排序 ===")
        for idx, node in enumerate(self.nodes[:3]):  # 输出Top3
            print(f"{idx+1}. {node.node_id} | 得分={node.root_cause_score} | 层级={LAYER_NAMES[node.layer]}")
        top_root_cause = self.nodes[0]
        ColorPrint.print_success(f"🏆 最终根因：{top_root_cause.node_id} | 描述：{top_root_cause.node_description}")
        
        # 6. 完成
        ColorPrint.print_step(5, "BCPN模型训练完成")

# ==============================================
# 【主程序】
# ==============================================
if __name__ == "__main__":
    np.random.seed(666)  # 固定随机种子，保证结果可复现
    
    print("="*70)
    print("BCPN 模型（适配SSD故障+重试风暴场景）")
    print("="*70)
    
    # 加载数据（适配generate_bcpn_data.py的输出路径）
    scenario_json_path = "bcpn_test_scenario/scenario_metadata.json"
    nodes, scenario_info, logs = ProductionDataLoader.load_from_multi_csv(scenario_json_path)
    
    # 训练BCPN模型
    if nodes:
        model = BCPNModel()
        model.fit(nodes, scenario_info, logs)
    else:
        ColorPrint.print_fail("无可用节点数据，程序退出")
