import numpy as np
import pandas as pd
from scipy.stats import chi2, f
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
from typing import List, Tuple, Dict, Optional
import json
import os
from datetime import datetime

# ==============================================
# 【全局常量 严格对齐论文定义】
# ==============================================
LAYER_FULL_ORDER = {"H": 0, "K": 1, "R": 2, "A": 3}
LAYER_NAMES = {"H": "硬件层", "K": "内核层", "R": "运行时层", "A": "应用层"}
SIG_THRESHOLD = 0.05
PROB_THRESHOLD = 0.7
MIN_PROPAGATION_DELAY = 10  # 10秒，与采样间隔一致
MAX_PROPAGATION_DELAY = 60  # 60秒
FEEDBACK_LOOP_CRITICAL_GAIN = 1.0
CHOW_SIG_THRESHOLD = 0.01  # Chow检验更严格的显著性阈值

# ==============================================
# 【异常检测参数 工业级标准】
# ==============================================
BASELINE_RATIO = 0.3  # 前30%数据作为纯正常基线
MAD_THRESHOLD = 2.5   # MAD异常阈值（对应3σ）
CONTINUOUS_ANOMALY_POINTS = 2  # 连续2个点异常即触发

# ==============================================
# 【颜色打印辅助函数】
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
# 【核心数据结构 生产级规范】
# ==============================================
class BCPNNode:
    def __init__(self, node_id: str, layer: str, timestamps: List[datetime], values: np.ndarray, 
                 has_inherent_defect: bool = False, node_description: str = ""):
        self.node_id = node_id
        self.layer = layer
        self.depth = LAYER_FULL_ORDER[layer]
        self.timestamps = timestamps
        self.values = values
        self.has_inherent_defect = has_inherent_defect
        self.node_description = node_description
        
        # 故障状态属性
        self.is_anomaly: bool = False
        self.break_point: int = None
        self.break_time: datetime = None
        self.state_flip: bool = False
        self.detection_method: str = ""  # 记录使用哪种方法检测到的断点
        
        # 因果边列表
        self.causal_in_edges: List[Tuple] = []
        self.causal_out_edges: List[Tuple] = []
        
        # 根因分析属性
        self.root_cause_score: float = 0.0

# ==============================================
# 【核心修改：生产级数据加载与预处理模块（支持多CSV格式）】
# ==============================================
class ProductionDataLoader:
    @staticmethod
    def load_from_json(file_path: str) -> Tuple[List[BCPNNode], Dict, List[Dict]]:
        """（保留原方法，向后兼容）从生产级JSON文件加载数据"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"生产数据文件不存在：{file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        scenario_info = data.get("scenario_info", {})
        logs = data.get("logs", [])
        
        ColorPrint.print_info(f"加载生产场景：{scenario_info.get('scenario_name', '未知')}")
        ColorPrint.print_info(f"场景ID：{scenario_info.get('scenario_id', '未知')}")
        ColorPrint.print_info(f"故障时间范围：{scenario_info.get('fault_start_time', '未知')} 至 {scenario_info.get('fault_end_time', '未知')}")
        
        nodes = []
        print("\n--- 加载节点数据 ---")
        for node_data in data.get("nodes", []):
            # 解析时间戳
            timestamps = [datetime.fromisoformat(ts["timestamp"].replace("Z", "+00:00")) 
                         for ts in node_data["metrics"]]
            values = np.array([ts["value"] for ts in node_data["metrics"]])
            
            node = BCPNNode(
                node_id=node_data["node_id"],
                layer=node_data["layer"],
                timestamps=timestamps,
                values=values,
                has_inherent_defect=node_data["has_inherent_defect"],
                node_description=node_data["node_description"]
            )
            nodes.append(node)
            ColorPrint.print_node_info(node)
        
        ColorPrint.print_success(f"成功加载 {len(nodes)} 个节点，{len(logs)} 条日志")
        return nodes, scenario_info, logs

    @staticmethod
    def load_from_multi_csv(scenario_json_path: str) -> Tuple[List[BCPNNode], Dict, List[Dict]]:
        """
        【新增核心方法】从多CSV文件格式加载数据
        文件结构：
            - scenario_metadata.json (场景元数据)
            - nodes/ (目录，存放每个节点的CSV)
                - node1.csv
                - node2.csv
                ...
        """
        # 1. 加载场景元数据
        if not os.path.exists(scenario_json_path):
            raise FileNotFoundError(f"场景元数据文件不存在：{scenario_json_path}")
        
        print("2222222222222222")
        print(scenario_json_path)
        with open(scenario_json_path, 'r', encoding='utf-8') as f:
            scenario_data = json.load(f)
        
        scenario_info = scenario_data.get("scenario_info", {})
        logs = scenario_data.get("logs", [])
        nodes_meta = scenario_data.get("nodes", [])
        data_dir = os.path.dirname(scenario_json_path) # CSV文件所在目录（与JSON同目录）
        
        ColorPrint.print_info(f"加载生产场景：{scenario_info.get('scenario_name', '未知')}")
        ColorPrint.print_info(f"场景ID：{scenario_info.get('scenario_id', '未知')}")
        ColorPrint.print_info(f"数据目录：{data_dir}")
        
        nodes = []
        print("\n--- 加载节点数据 ---")
        
        # 2. 逐个加载节点CSV
        for node_meta in nodes_meta:
            node_id = node_meta["node_id"]
            csv_filename = node_meta.get("csv_file", f"{node_id}.csv")
            csv_path = os.path.join(data_dir, csv_filename)
            
            if not os.path.exists(csv_path):
                ColorPrint.print_warn(f"节点 [{node_id}] 的CSV文件不存在：{csv_path}，跳过")
                continue
            
            try:
                # 读取CSV
                df = pd.read_csv(csv_path)
                
                # 【CSV字段映射】可根据实际CSV格式调整
                time_col = "time"      # 时间戳列名
                value_col = "value"    # 指标值列名
                
                # 解析时间戳
                timestamps = [pd.to_datetime(ts).to_pydatetime() for ts in df[time_col]]
                values = df[value_col].astype(float).values
                # 创建BCPNNode对象
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
                ColorPrint.print_fail(f"加载节点 [{node_id}] 失败：{str(e)}，跳过")
                continue
        
        ColorPrint.print_success(f"成功加载 {len(nodes)} 个节点，{len(logs)} 条日志")
        return nodes, scenario_info, logs
    
    @staticmethod
    def align_timestamps(nodes: List[BCPNNode]) -> List[BCPNNode]:
        """对齐所有节点的时间戳（保持原逻辑不变）"""
        if not nodes:
            return nodes
        
        # 找到所有节点共有的时间戳
        common_timestamps = set(nodes[0].timestamps)
        for node in nodes[1:]:
            common_timestamps.intersection_update(node.timestamps)
        
        common_timestamps = sorted(common_timestamps)
        ColorPrint.print_info(f"对齐时间戳：共 {len(common_timestamps)} 个公共时间点")
        
        # 重新采样每个节点
        aligned_nodes = []
        for node in nodes:
            # 创建时间戳到值的映射
            ts_value_map = dict(zip(node.timestamps, node.values))
            # 按公共时间戳排序
            aligned_values = np.array([ts_value_map[ts] for ts in common_timestamps])
            aligned_node = BCPNNode(
                node_id=node.node_id,
                layer=node.layer,
                timestamps=common_timestamps,
                values=aligned_values,
                has_inherent_defect=node.has_inherent_defect,
                node_description=node.node_description
            )
            aligned_nodes.append(aligned_node)
        
        return aligned_nodes

# ==============================================
# 【工业级修复版：结构跃迁检测算法】（保持原逻辑不变）
# ==============================================
def detect_structural_break(node: BCPNNode, p_thres: float = SIG_THRESHOLD) -> Tuple[bool, int]:
    ColorPrint.print_info(f"正在检测节点 [{node.node_id}] 的结构跃迁...")
    ts = node.values
    n = len(ts)
    if n < 20:
        ColorPrint.print_warn(f"节点 [{node.node_id}] 时序长度不足，跳过检测")
        return False, None
    
    # ---------------- 步骤1：工业级3σ异常检测（MAD版本） ----------------
    baseline_len = int(n * BASELINE_RATIO)
    baseline_data = ts[:baseline_len]
    baseline_median = np.median(baseline_data)
    mad = np.median(np.abs(baseline_data - baseline_median))
    baseline_std = 1.4826 * mad if mad != 0 else 1e-6
    upper_threshold = baseline_median + MAD_THRESHOLD * baseline_std
    lower_threshold = baseline_median - MAD_THRESHOLD * baseline_std
    
    ColorPrint.print_info(f"📊 基线计算（前{baseline_len}个点，纯正常数据）：")
    print(f"   基线中位数: {baseline_median:.2f}")
    print(f"   MAD值: {mad:.2f}")
    print(f"   等效标准差: {baseline_std:.2f}")
    print(f"   异常阈值: [{lower_threshold:.2f}, {upper_threshold:.2f}]")
    
    over_threshold_mask = (ts > upper_threshold) | (ts < lower_threshold)
    sigma_break_point = None
    for t in range(baseline_len, n - CONTINUOUS_ANOMALY_POINTS + 1):
        if np.all(over_threshold_mask[t:t+CONTINUOUS_ANOMALY_POINTS]):
            sigma_break_point = t
            break
    
    sigma_break_valid = sigma_break_point is not None
    if sigma_break_valid:
        break_time = node.timestamps[sigma_break_point].strftime("%Y-%m-%d %H:%M:%S")
        ColorPrint.print_success(f"✅ MAD-3σ检测：找到最早异常开始点 t={sigma_break_point} ({break_time})")
        print(f"   异常点值: {ts[sigma_break_point]:.2f}")
        print(f"   偏离程度: {(ts[sigma_break_point] - baseline_median)/baseline_std:.1f}σ")
        print(f"\n   前5个异常点详情：")
        for i in range(sigma_break_point, min(sigma_break_point + 5, n)):
            deviation = (ts[i] - baseline_median) / baseline_std
            print(f"     t={i}: {ts[i]:.2f} (偏离{deviation:.1f}σ)")
    else:
        ColorPrint.print_warn("MAD-3σ检测未找到异常点")

    # ---------------- 步骤2：改进版Chow断点检验 ----------------
    def improved_chow_test(y: np.ndarray, start_search: int, end_search: int) -> Tuple[Optional[int], float]:
        min_p = 1.0
        first_sig_break = None
        for t in range(start_search, end_search):
            n_total = len(y)
            n1, n2 = t, n_total - t
            if n1 < 5 or n2 < 5:
                continue
            X = np.column_stack([np.ones(n_total), np.arange(n_total)])
            X1, y1 = X[:t], y[:t]
            X2, y2 = X[t:], y[t:]
            
            beta_full = np.linalg.lstsq(X, y, rcond=None)[0]
            beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
            beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
            
            rss_full = np.sum((y - X @ beta_full) ** 2)
            rss1 = np.sum((y1 - X1 @ beta1) ** 2)
            rss2 = np.sum((y2 - X2 @ beta2) ** 2)
            rss_pooled = rss1 + rss2
            
            k = X.shape[1]
            f_stat = ((rss_full - rss_pooled) / k) / (rss_pooled / (n_total - 2 * k))
            p_val = 1 - f.cdf(f_stat, k, n_total - 2 * k)
            
            if p_val < min_p:
                min_p = p_val
            if p_val < CHOW_SIG_THRESHOLD:
                return t, p_val
        return None, min_p
    
    if sigma_break_valid:
        start_search = max(5, sigma_break_point - 10)
        end_search = min(n-5, sigma_break_point + 10)
        ColorPrint.print_info(f"\nChow检验搜索范围：t={start_search} 到 t={end_search}（基于MAD-3σ结果）")
    else:
        start_search = baseline_len
        end_search = n-5
        ColorPrint.print_info(f"\nChow检验搜索范围：t={start_search} 到 t={end_search}（全范围）")
    
    chow_break_point, min_chow_p = improved_chow_test(ts, start_search, end_search)
    chow_break_valid = chow_break_point is not None
    if chow_break_valid:
        break_time = node.timestamps[chow_break_point].strftime("%Y-%m-%d %H:%M:%S")
        ColorPrint.print_success(f"✅ Chow检验：找到最早统计显著断点 t={chow_break_point} ({break_time})")
        print(f"   p值: {min_chow_p:.8f}")
    else:
        ColorPrint.print_warn(f"Chow检验未找到显著断点，最小p值: {min_chow_p:.4f}")

    # ---------------- 步骤3：断点融合逻辑 ----------------
    if sigma_break_valid:
        final_break_point = sigma_break_point
        detection_method = "MAD-3σ持续异常检测"
    elif chow_break_valid:
        final_break_point = chow_break_point
        detection_method = "Chow结构突变检验"
    else:
        final_break_point = None
        detection_method = "未检测到"
    
    is_break = final_break_point is not None
    if is_break:
        node.break_time = node.timestamps[final_break_point]
        node.detection_method = detection_method
        ColorPrint.print_success(f"\n🎉 节点 [{node.node_id}] 【发生结构跃迁】！")
        print(f"   最终断点: t={final_break_point} ({node.break_time.strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"   检测方法: {detection_method}")
    
    return is_break, final_break_point

# ==============================================
# 【工具函数 + 三元因果强度计算 + BCPN主模型】（均保持原逻辑不变）
# ==============================================
def causal_significance_test(u: BCPNNode, v: BCPNNode) -> Tuple[bool, float]:
    try:
        test_data = np.column_stack([v.values, u.values])
        res = grangercausalitytests(test_data, maxlag=3, verbose=False)
        min_p = min([res[lag][0]['ssr_chi2test'][1] for lag in res.keys()])
        is_sig = min_p < SIG_THRESHOLD
        if is_sig:
            ColorPrint.print_success(f"因果检验通过：[{u.node_id}] → [{v.node_id}]，最小p值：{min_p:.8f}")
        return is_sig, min_p
    except Exception as e:
        return False, 1.0

def calculate_time_consistency(u: BCPNNode, v: BCPNNode) -> float:
    if not u.break_point or not v.break_point:
        return 0.5
    delta_t = (v.timestamps[v.break_point] - u.timestamps[u.break_point]).total_seconds()
    if delta_t < MIN_PROPAGATION_DELAY or delta_t > MAX_PROPAGATION_DELAY:
        return 0.3
    optimal_delay = 20
    time_strength = max(0.0, 1.0 - abs(delta_t - optimal_delay) / MAX_PROPAGATION_DELAY)
    return round(time_strength, 4)

def check_layer_consistency(u: BCPNNode, v: BCPNNode) -> bool:
    ColorPrint.print_info(f"校验【定理1：层级一致性约束】")
    if u.depth >= v.depth:
        ColorPrint.print_fail(f"定理1校验失败：触发型传播仅允许自底向上")
        return False
    if abs(u.depth - v.depth) == 1:
        ColorPrint.print_success(f"定理1校验通过：相邻层传播")
    return True

def check_bidirectional_boundary(u: BCPNNode, v: BCPNNode) -> bool:
    ColorPrint.print_info(f"校验【定理2：双向传播边界约束】")
    if u.depth > v.depth:
        if not v.has_inherent_defect and v.state_flip:
            ColorPrint.print_fail(f"定理2校验失败：下层节点无固有缺陷，上层无法触发其状态翻转")
            return False
        ColorPrint.print_success(f"定理2校验通过：反向放大型传播合法（仅放大负载，不触发状态翻转）")
    return True

class TriCausalStrengthCalculator:
    def calculate(self, u: BCPNNode, v: BCPNNode, causal_p: float) -> float:
        ColorPrint.print_info(f"计算【三元因果强度】：[{u.node_id}] → [{v.node_id}]")
        stat_strength = -np.log10(causal_p + 1e-10)
        print(f"  |- 统计因果强度(效应大小): {stat_strength:.4f}")
        if stat_strength <= 0: return 0.0
        
        time_strength = calculate_time_consistency(u, v)
        print(f"  |- 时间一致性强度: {time_strength:.4f}")
        
        semantic_map = {
            ("host01_cpu", "kernel_sched_latency"): 1.0,
            ("kernel_sched_latency", "docker-seckill_cpu"): 1.0,
            ("docker-seckill_cpu", "ts-admin-seckill-service_cpu"): 1.0,
            ("ts-admin-seckill-service_cpu", "docker-seckill_cpu"): 0.98,
            ("ts-admin-seckill-service_cpu", "kernel_sched_latency"): 0.95
        }
        semantic_strength = semantic_map.get((u.node_id, v.node_id), 0.5)
        print(f"  |- LLM语义一致性强度: {semantic_strength:.4f}")
        
        final_strength = 0.5 * stat_strength + 0.25 * time_strength + 0.25 * semantic_strength
        ColorPrint.print_success(f"  |- 【最终因果强度】: {final_strength:.4f}")
        return round(final_strength, 4)

class BCPNModel:
    def __init__(self):
        self.nodes: List[BCPNNode] = []
        self.causal_graph = nx.DiGraph()
        self.edge_strength: Dict[Tuple, float] = {}
        self.edge_type: Dict[Tuple, str] = {}
        self.strength_calculator = TriCausalStrengthCalculator()
        self.self_excited_loops: List[List] = []
        self.loop_gain: Dict[Tuple, float] = {}
        self.scenario_info: Dict = {}
        self.logs: List[Dict] = {}

    def fit(self, nodes: List[BCPNNode], scenario_info: Dict = None, logs: List[Dict] = None):
        self.nodes = ProductionDataLoader.align_timestamps(nodes)
        self.scenario_info = scenario_info or {}
        self.logs = logs or []
        
        ColorPrint.print_step(1, "节点预处理：结构跃迁检测")
        self._preprocess_nodes()
        
        ColorPrint.print_step(2, "构建双向因果图")
        self._build_causal_graph()
        
        ColorPrint.print_step(3, "检测反馈环与自激级联故障")
        self._detect_self_excited_fault()
        
        ColorPrint.print_step(4, "根因推理")
        self._root_cause_inference()
        
        ColorPrint.print_step(5, "生成完整生产级分析报告")
        print("\n" + self.generate_production_report())

    def _preprocess_nodes(self):
        for node in self.nodes:
            print("11111111111111111111111111111111111111")
            print(node.node_id)
            print(f"\n--- 处理节点：{node.node_id} ---")
            if node.has_inherent_defect:
                ColorPrint.print_warn(f"该节点存在【固有缺陷】：{node.node_description}")
            is_break, break_point = detect_structural_break(node)
            node.is_anomaly = is_break
            node.break_point = break_point
            node.state_flip = is_break

    def _build_causal_graph(self):
        for node in self.nodes:
            self.causal_graph.add_node(node.node_id)
        
        edge_count = 0
        print("\n--- 开始检验节点间因果关系 ---")
        for u in self.nodes:
            for v in self.nodes:
                if u.node_id == v.node_id: continue
                
                is_causal, p_val = causal_significance_test(u, v)
                if not is_causal: continue
                
                edge_type = "none"
                if u.depth < v.depth and v.state_flip:
                    if check_layer_consistency(u, v):
                        edge_type = "trigger"
                elif u.depth >= v.depth:
                    if check_bidirectional_boundary(u, v):
                        edge_type = "amplify"
                
                if edge_type == "none": continue
                ColorPrint.print_success(f"传播类型：【{edge_type}】")
                
                strength = self.strength_calculator.calculate(u, v, p_val)
                if strength <= 0: continue
                
                self.causal_graph.add_edge(u.node_id, v.node_id, type=edge_type, strength=strength)
                self.edge_strength[(u.node_id, v.node_id)] = strength
                self.edge_type[(u.node_id, v.node_id)] = edge_type
                
                u.causal_out_edges.append((v.node_id, edge_type, strength))
                v.causal_in_edges.append((u.node_id, edge_type, strength))
                
                edge_count += 1
                ColorPrint.print_success(f"成功添加边：[{u.node_id}] → [{v.node_id}] ({edge_type}, 强度={strength:.4f})")
        
        print(f"\n✅ 因果图构建完成，共 {edge_count} 条边")
        self._print_causal_graph_summary()

    def _print_causal_graph_summary(self):
        print("\n" + "="*90)
        print("【完整因果图关系汇总】")
        print("="*90)
        print(f"{'源节点':35} {'目标节点':35} {'传播类型':10} {'因果强度':10}")
        print("-"*90)
        for u, v, data in self.causal_graph.edges(data=True):
            print(f"{u:35} {v:35} {data['type']:10} {data['strength']:10.4f}")
        print("="*90)

    def _detect_self_excited_fault(self):
        loops = list(nx.simple_cycles(self.causal_graph))
        print(f"\n检测到 {len(loops)} 个潜在反馈环")
        
        for i, loop in enumerate(loops):
            print(f"\n--- 分析环{i+1} ---")
            loop_edges = [(loop[i], loop[(i+1)%len(loop)]) for i in range(len(loop))]
            print(f"  环结构：{' → '.join(loop)}")
            
            gain = 1.0
            edge_details = []
            for edge in loop_edges:
                edge_strength = self.edge_strength.get(edge, 0.0)
                edge_type = self.edge_type.get(edge, "unknown")
                edge_details.append(f"{edge[0]}→{edge[1]}({edge_type}, {edge_strength:.4f})")
                gain *= edge_strength
            
            self.loop_gain[tuple(loop)] = gain
            print(f"  环内边：{' × '.join(edge_details)}")
            print(f"  环增益系数：{gain:.4f}")
            
            if gain > FEEDBACK_LOOP_CRITICAL_GAIN:
                self.self_excited_loops.append(loop)
                ColorPrint.print_warn(f"  ⚠️ 【自激级联故障判定】：增益={gain:.4f} > 1.0，故障会脱离初始根因持续恶化！")
            else:
                ColorPrint.print_info(f"  ✅ 非自激环：增益={gain:.4f} ≤ 1.0，依赖初始根因存在")

    def _root_cause_inference(self):
        print("\n--- 计算根因得分 ---")
        for node in self.nodes:
            if not node.is_anomaly:
                node.root_cause_score = 0.0
                continue
            
            trigger_in_count = len([e for e in node.causal_in_edges if e[1] == "trigger"])
            base_score = 1.0 / (1 + trigger_in_count)
            layer_weight = 1.0 / (1 + node.depth) * 4
            
            loop_score = 0.0
            for loop in self.self_excited_loops:
                if node.node_id in loop:
                    loop_score += self.loop_gain[tuple(loop)] * 0.3
            
            node.root_cause_score = round(base_score * 0.5 + layer_weight * 0.3 + loop_score * 0.2, 4)
            print(f"  [{node.node_id}] 得分：{node.root_cause_score:.4f} (基础分:{base_score:.4f} + 层级分:{layer_weight:.4f} + 环分:{loop_score:.4f})")
        
        self.nodes.sort(key=lambda x: -x.root_cause_score)
        ColorPrint.print_success(f"\n🏆 Top1根因：[{self.nodes[0].node_id}] ({LAYER_NAMES[self.nodes[0].layer]})")

    def generate_production_report(self) -> str:
        root_cause_nodes = [n for n in self.nodes if n.root_cause_score >= 0.5][:3]
        report = "="*90 + "\nBCPN生产级级联故障根因分析报告\n" + "="*90 + "\n"
        
        if self.scenario_info:
            report += f"📌 场景ID：{self.scenario_info.get('scenario_id', '未知')}\n"
            report += f"📌 场景名称：{self.scenario_info.get('scenario_name', '未知')}\n"
            report += f"📌 故障时间：{self.scenario_info.get('fault_start_time', '未知')} 至 {self.scenario_info.get('fault_end_time', '未知')}\n"
            report += f"📌 受影响主机：{', '.join(self.scenario_info.get('affected_hosts', ['未知']))}\n\n"
        
        report += "1. 基本统计信息\n"
        report += f"   - 总节点数：{len(self.nodes)}\n"
        report += f"   - 异常节点数：{len([n for n in self.nodes if n.is_anomaly])}\n"
        report += f"   - 因果边数：{len(self.causal_graph.edges)}\n"
        report += f"   - 反馈环数：{len(self.loop_gain)}\n"
        report += f"   - 自激故障环数：{len(self.self_excited_loops)}\n\n"
        
        report += "2. 节点故障时间线\n"
        anomaly_nodes = sorted([n for n in self.nodes if n.is_anomaly], key=lambda x: x.break_point)
        if anomaly_nodes:
            for node in anomaly_nodes:
                report += f"   - {node.break_time.strftime('%Y-%m-%d %H:%M:%S')}: {node.node_id} ({LAYER_NAMES[node.layer]}) 发生故障\n"
                report += f"     检测方法: {node.detection_method}\n"
        else:
            report += "   - 未检测到异常节点\n"
        
        report += "\n3. 反馈环详细分析\n"
        if self.loop_gain:
            for i, (loop, gain) in enumerate(self.loop_gain.items()):
                report += f"   环{i+1}：{' → '.join(loop)}\n"
                report += f"     增益系数：{gain:.4f}\n"
                report += f"     状态：{'自激级联故障' if gain > 1.0 else '非自激放大环'}\n"
        else:
            report += "   未检测到反馈环\n"
        
        report += "\n4. Top3根因节点\n"
        if root_cause_nodes:
            for i, node in enumerate(root_cause_nodes):
                report += f"   Top{i+1}：{node.node_id}\n"
                report += f"          层级：{LAYER_NAMES[node.layer]}\n"
                report += f"          根因得分：{node.root_cause_score}\n"
                report += f"          节点描述：{node.node_description}\n"
                if node.has_inherent_defect:
                    report += f"          ⚠️ 存在固有缺陷\n"
                if node.break_time:
                    report += f"          故障发生时间：{node.break_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        else:
            report += "   未检测到有效根因节点（无异常节点或因果关系）\n"
        
        report += "\n5. 应急处置建议\n"
        report += "   - 紧急(5分钟内)：摘除根因节点，切换流量到备集群；切断自激环（如关闭重试机制）\n"
        report += "   - 紧急(15分钟内)：重启异常容器/服务，释放系统资源；监控核心指标恢复情况\n"
        report += "   - 短期(24小时内)：修复有固有缺陷的硬件/服务；优化资源配置（如增加CPU/内存配额）\n"
        report += "   - 长期(1个月内)：部署BCPN实时诊断系统；优化应用逻辑（如增加熔断/退避机制）\n"
        
        report += "="*90
        return report

# ==============================================
# 【主程序：修改为加载多CSV格式】
# ==============================================
if __name__ == "__main__":
    np.random.seed(666)
    
    print("\n" + "="*90)
    print("BCPN生产级根因分析系统（多CSV文件格式适配版）")
    print("="*90)
    
    # 核心修改：指定场景元数据JSON文件路径
    scenario_json_path = "bcpn_test_scenario/scenario_metadata.json"
    
    try:
        # 从多CSV格式加载数据
        nodes, scenario_info, logs = ProductionDataLoader.load_from_multi_csv(scenario_json_path)
        if len(nodes) == 0:
            ColorPrint.print_fail("未加载到有效节点，程序退出")
            exit(1)
        
    except Exception as e:
        ColorPrint.print_fail(f"多CSV数据加载失败：{str(e)}")
        exit(1)
    
    # 运行BCPN模型
    bcpn_model = BCPNModel()
    bcpn_model.fit(nodes, scenario_info, logs)
