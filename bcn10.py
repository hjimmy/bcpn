import numpy as np
import pandas as pd
from scipy.stats import chi2, f, pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import json
import os
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
import sys
from tqdm import tqdm

# ==============================================
# 【全局配置类 生产级规范】
# ==============================================
@dataclass
class BCPNConfig:
    """BCPN模型全局配置，严格对齐论文定义"""
    # 层级定义
    LAYER_FULL_ORDER: Dict[str, int] = None
    LAYER_NAMES: Dict[str, str] = None
    
    # 统计显著性阈值
    SIG_THRESHOLD: float = 0.05
    PROB_THRESHOLD: float = 0.7
    CHOW_SIG_THRESHOLD: float = 0.01
    
    # 正向单调变化验证阈值
    MIN_POSITIVE_CORR: float = 0.3
    CORR_SIG_THRESHOLD: float = 0.05
    
    # 传播延迟参数（区分trigger和amplify）
    MIN_TRIGGER_DELAY: int = 5    # 触发型最小传播延迟（秒）
    MAX_TRIGGER_DELAY: int = 60   # 触发型最大传播延迟（秒）
    OPTIMAL_TRIGGER_DELAY: int = 20  # 触发型最优传播延迟（秒）
    
    MIN_AMPLIFY_DELAY: int = 0    # 放大型最小传播延迟（秒）
    MAX_AMPLIFY_DELAY: int = 120  # 放大型最大传播延迟（秒）
    OPTIMAL_AMPLIFY_DELAY: int = 30  # 放大型最优传播延迟（秒）
    
    # 采样间隔（用于正反馈环检测）
    SAMPLING_INTERVAL: int = 10   # 数据采样间隔（秒）
    
    # 反馈环参数
    FEEDBACK_LOOP_CRITICAL_GAIN: float = 1.0
    MIN_LOOP_GAIN_TO_REPORT: float = 0.1  # 只报告增益大于此值的环
    
    # 异常检测参数
    BASELINE_RATIO: float = 0.3
    MAD_THRESHOLD: float = 2.5
    CONTINUOUS_ANOMALY_POINTS: int = 2
    
    # 三元因果强度权重(论文3.3.3节)
    STAT_WEIGHT: float = 0.4
    TEMPORAL_WEIGHT: float = 0.35
    SEMANTIC_WEIGHT: float = 0.25
    
    # 因果强度阈值（过滤弱边）
    MIN_CAUSAL_STRENGTH: float = 0.1
    
    # 放大型传播衰减系数(论文3.4.1节)
    AMPLIFICATION_DECAY: float = 0.7
    
    # LLM配置
    SILICONFLOW_API_KEY: str = None
    SILICONFLOW_API_URL: str = "https://api.siliconflow.cn/v1/chat/completions"
    LLM_MODEL: str = "Qwen/Qwen2.5-72B-Instruct"
    LLM_TIMEOUT: int = 30
    LLM_MAX_RETRIES: int = 3
    
    # 日志配置
    LOG_LEVEL: int = logging.INFO
    LOG_FILE: Optional[str] = None

    def __post_init__(self):
        if self.LAYER_FULL_ORDER is None:
            self.LAYER_FULL_ORDER = {"H": 0, "K": 1, "R": 2, "A": 3}
        if self.LAYER_NAMES is None:
            self.LAYER_NAMES = {"H": "硬件层", "K": "内核层", "R": "运行时层", "A": "应用层"}
        if self.SILICONFLOW_API_KEY is None:
            self.SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")

# 全局配置实例
config = BCPNConfig()

# ==============================================
# 【日志系统初始化】
# ==============================================
def setup_logging():
    """配置生产级日志系统"""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if config.LOG_FILE:
        handlers.append(logging.FileHandler(config.LOG_FILE, encoding="utf-8"))
    
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=log_format,
        handlers=handlers
    )
    
    # 屏蔽第三方库的冗余日志
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("statsmodels").setLevel(logging.WARNING)

setup_logging()
logger = logging.getLogger(__name__)

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
    def print_step(step_num: int, title: str):
        msg = f"\n{ColorPrint.HEADER}{ColorPrint.BOLD}===== 步骤 {step_num}：{title} ====={ColorPrint.ENDC}"
        print(msg)
        logger.info(f"步骤 {step_num}：{title}")
    
    @staticmethod
    def print_info(msg: str):
        print(f"{ColorPrint.CYAN}[信息]{ColorPrint.ENDC} {msg}")
        logger.info(msg)
    
    @staticmethod
    def print_success(msg: str):
        print(f"{ColorPrint.GREEN}[成功]{ColorPrint.ENDC} {msg}")
        logger.info(msg)
    
    @staticmethod
    def print_warn(msg: str):
        print(f"{ColorPrint.WARNING}[警告]{ColorPrint.ENDC} {msg}")
        logger.warning(msg)
    
    @staticmethod
    def print_fail(msg: str):
        print(f"{ColorPrint.FAIL}[失败]{ColorPrint.ENDC} {msg}")
        logger.error(msg)
    
    @staticmethod
    def print_node_info(node: 'BCPNNode'):
        print(f"  {ColorPrint.BOLD}节点ID:{ColorPrint.ENDC} {node.node_id:35} | "
              f"{ColorPrint.BOLD}层级:{ColorPrint.ENDC} {config.LAYER_NAMES[node.layer]:8} | "
              f"{ColorPrint.BOLD}固有缺陷:{ColorPrint.ENDC} {'是' if node.has_inherent_defect else '否':3}")
        print(f"    {ColorPrint.BOLD}描述:{ColorPrint.ENDC} {node.node_description}")

# ==============================================
# 【核心数据结构 生产级规范】
# ==============================================
class BCPNNode:
    """BCPN节点数据结构，严格对齐论文3.2.2节定义"""
    def __init__(self, node_id: str, layer: str, timestamps: List[datetime], values: np.ndarray, 
                 has_inherent_defect: bool = False, node_description: str = ""):
        # 输入验证
        if layer not in config.LAYER_FULL_ORDER:
            raise ValueError(f"无效层级: {layer}，必须是H/K/R/A之一")
        if len(timestamps) != len(values):
            raise ValueError(f"时间戳长度({len(timestamps)})与值长度({len(values)})不匹配")
        
        self.node_id = node_id
        self.layer = layer
        self.depth = config.LAYER_FULL_ORDER[layer]
        self.timestamps = np.array(timestamps)
        self.values = np.array(values, dtype=np.float64)
        self.has_inherent_defect = has_inherent_defect
        self.node_description = node_description
        
        # 异常检测结果
        self.is_anomaly: bool = False
        self.break_point: Optional[int] = None
        self.break_time: Optional[datetime] = None
        self.state_flip: bool = False
        self.detection_method: str = ""
        
        # 因果图结构
        self.causal_in_edges: List[Tuple[str, str, float]] = []
        self.causal_out_edges: List[Tuple[str, str, float]] = []
        
        # 根因评分
        self.root_cause_score: float = 0.0
        self.prior_prob: float = 0.0
        self.likelihood: float = 0.0

    def __repr__(self) -> str:
        return f"BCPNNode(id={self.node_id}, layer={self.layer}, anomaly={self.is_anomaly})"

# ==============================================
# 【HTTP请求会话 带重试机制】
# ==============================================
def create_retry_session() -> requests.Session:
    """创建带自动重试的HTTP会话"""
    session = requests.Session()
    retry_strategy = Retry(
        total=config.LLM_MAX_RETRIES,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

http_session = create_retry_session()

# ==============================================
# 【生产级数据加载 优化版】
# ==============================================
class ProductionDataLoader:
    @staticmethod
    def load_from_json(file_path: str) -> Tuple[List[BCPNNode], Dict, List[Dict]]:
        """从JSON文件加载生产数据"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"生产数据文件不存在：{file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        scenario_info = data.get("scenario_info", {})
        logs = data.get("logs", [])
        
        ColorPrint.print_info(f"加载生产场景：{scenario_info.get('scenario_name', '未知')}")
        ColorPrint.print_info(f"场景ID：{scenario_info.get('scenario_id', '未知')}")
        
        nodes = []
        print("\n--- 加载节点数据 ---")
        for node_data in tqdm(data.get("nodes", []), desc="加载节点"):
            try:
                timestamps = [datetime.fromisoformat(ts["timestamp"].replace("Z", "+00:00")) 
                             for ts in node_data["metrics"]]
                values = np.array([ts["value"] for ts in node_data["metrics"]])
                
                node = BCPNNode(
                    node_id=node_data["node_id"],
                    layer=node_data["layer"],
                    timestamps=timestamps,
                    values=values,
                    has_inherent_defect=node_data.get("has_inherent_defect", False),
                    node_description=node_data.get("node_description", "")
                )
                nodes.append(node)
                ColorPrint.print_node_info(node)
            except Exception as e:
                ColorPrint.print_fail(f"加载节点失败：{str(e)}")
                continue
        
        ColorPrint.print_success(f"成功加载 {len(nodes)} 个节点，{len(logs)} 条日志")
        return nodes, scenario_info, logs

    @staticmethod
    def load_from_multi_csv(scenario_json_path: str) -> Tuple[List[BCPNNode], Dict, List[Dict]]:
        """从多个CSV文件加载生产数据"""
        if not os.path.exists(scenario_json_path):
            raise FileNotFoundError(f"场景元数据文件不存在：{scenario_json_path}")
        
        with open(scenario_json_path, 'r', encoding='utf-8') as f:
            scenario_data = json.load(f)
        
        scenario_info = scenario_data.get("scenario_info", {})
        logs = scenario_data.get("logs", [])
        nodes_meta = scenario_data.get("nodes", [])
        data_dir = os.path.dirname(scenario_json_path)
        
        ColorPrint.print_info(f"加载生产场景：{scenario_info.get('scenario_name', '未知')}")
        ColorPrint.print_info(f"数据目录：{data_dir}")
        
        nodes = []
        for node_meta in tqdm(nodes_meta, desc="加载CSV节点"):
            node_id = node_meta["node_id"]
            csv_filename = node_meta.get("csv_file", f"{node_id}.csv")
            csv_path = os.path.join(data_dir, csv_filename)
            
            if not os.path.exists(csv_path):
                ColorPrint.print_warn(f"节点 [{node_id}] 文件不存在，跳过")
                continue
            
            try:
                df = pd.read_csv(csv_path)
                if "time" not in df.columns or "value" not in df.columns:
                    raise ValueError("CSV文件必须包含'time'和'value'列")
                
                timestamps = [pd.to_datetime(ts).to_pydatetime() for ts in df["time"]]
                values = df["value"].astype(float).values
                
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
                ColorPrint.print_fail(f"加载节点 [{node_id}] 失败：{str(e)}")
                continue
        
        ColorPrint.print_success(f"成功加载 {len(nodes)} 个节点")
        return nodes, scenario_info, logs
    
    @staticmethod
    def align_timestamps(nodes: List[BCPNNode]) -> List[BCPNNode]:
        """优化版时间戳对齐，使用pandas merge提高效率"""
        if not nodes:
            return nodes
        
        # 构建时间戳-值DataFrame
        dfs = []
        for node in nodes:
            df = pd.DataFrame({
                "timestamp": node.timestamps,
                node.node_id: node.values
            })
            dfs.append(df)
        
        # 合并所有DataFrame，取交集
        merged_df = dfs[0]
        for df in dfs[1:]:
            merged_df = pd.merge(merged_df, df, on="timestamp", how="inner")
        
        common_timestamps = merged_df["timestamp"].tolist()
        ColorPrint.print_info(f"对齐时间戳：共 {len(common_timestamps)} 个公共时间点")
        
        # 生成对齐后的节点
        aligned_nodes = []
        for node in nodes:
            aligned_values = merged_df[node.node_id].values
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
# 【工业级结构跃迁检测 修复版】
# ==============================================
def detect_structural_break(node: BCPNNode, p_thres: float = config.SIG_THRESHOLD) -> Tuple[bool, Optional[int]]:
    """检测节点时序的结构跃迁，结合MAD-3σ与改进Chow检验"""
    ColorPrint.print_info(f"正在检测节点 [{node.node_id}] 的结构跃迁...")
    ts = node.values
    n = len(ts)
    
    if n < 20:
        ColorPrint.print_warn(f"节点时序长度不足({n}<20)，跳过检测")
        return False, None
    
    baseline_len = int(n * config.BASELINE_RATIO)
    baseline_data = ts[:baseline_len]
    
    # MAD-3σ异常检测
    baseline_median = np.median(baseline_data)
    mad = np.median(np.abs(baseline_data - baseline_median))
    baseline_std = 1.4826 * mad if mad != 0 else 1e-6
    upper_threshold = baseline_median + config.MAD_THRESHOLD * baseline_std
    lower_threshold = baseline_median - config.MAD_THRESHOLD * baseline_std
    
    # 向量化检测连续异常点
    over_threshold_mask = (ts > upper_threshold) | (ts < lower_threshold)
    # 使用滑动窗口求和检测连续异常
    window = np.ones(config.CONTINUOUS_ANOMALY_POINTS, dtype=int)
    continuous_anomaly = np.convolve(over_threshold_mask.astype(int), window, mode='valid')
    sigma_break_indices = np.where(continuous_anomaly == config.CONTINUOUS_ANOMALY_POINTS)[0]
    
    sigma_break_point = sigma_break_indices[0] + baseline_len if len(sigma_break_indices) > 0 else None
    sigma_break_valid = sigma_break_point is not None

    def improved_chow_test(y: np.ndarray, start_search: int, end_search: int) -> Tuple[Optional[int], float]:
        """改进的Chow检验，向量化计算提高效率"""
        min_p = 1.0
        first_sig_break = None
        
        # 预计算X矩阵
        X = np.column_stack([np.ones(len(y)), np.arange(len(y))])
        XTX = X.T @ X
        XTy = X.T @ y
        beta_full = np.linalg.solve(XTX, XTy)
        rss_full = np.sum((y - X @ beta_full) ** 2)
        k = X.shape[1]
        
        for t in range(start_search, end_search):
            n1, n2 = t, len(y) - t
            if n1 < 5 or n2 < 5:
                continue
            
            # 分段计算
            X1, X2 = X[:t], X[t:]
            y1, y2 = y[:t], y[t:]
            
            beta1 = np.linalg.solve(X1.T @ X1, X1.T @ y1)
            beta2 = np.linalg.solve(X2.T @ X2, X2.T @ y2)
            
            rss1 = np.sum((y1 - X1 @ beta1) ** 2)
            rss2 = np.sum((y2 - X2 @ beta2) ** 2)
            rss_pooled = rss1 + rss2
            
            f_stat = ((rss_full - rss_pooled) / k) / (rss_pooled / (len(y) - 2 * k))
            p_val = 1 - f.cdf(f_stat, k, len(y) - 2 * k)
            
            if p_val < min_p:
                min_p = p_val
            if p_val < config.CHOW_SIG_THRESHOLD:
                return t, p_val
        
        return None, min_p

    # 确定Chow检验搜索范围
    start_search = max(5, sigma_break_point - 10) if sigma_break_valid else baseline_len
    end_search = min(n-5, sigma_break_point + 10) if sigma_break_valid else n-5
    
    chow_break_point, _ = improved_chow_test(ts, start_search, end_search)
    chow_break_valid = chow_break_point is not None
    
    # 确定最终断点
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
    
    # 统一赋值所有状态跃迁属性
    if is_break and final_break_point is not None:
        node.break_point = final_break_point
        node.break_time = node.timestamps[final_break_point]
        node.detection_method = detection_method
        node.is_anomaly = True
        node.state_flip = True
        
        ColorPrint.print_success(
            f"节点 [{node.node_id}] 发生结构跃迁 | "
            f"时间：{node.break_time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"方法：{detection_method}"
        )
    else:
        # 重置所有异常属性
        node.break_point = None
        node.break_time = None
        node.detection_method = "未检测到"
        node.is_anomaly = False
        node.state_flip = False
        
        ColorPrint.print_info(f"节点 [{node.node_id}] 未检测到结构跃迁")
    
    return is_break, final_break_point

# ==============================================
# 【工具函数 新增正向单调变化验证】
# ==============================================
# Granger因果检验结果缓存
_granger_cache: Dict[Tuple[str, str], Tuple[bool, float]] = {}

def causal_significance_test(u: BCPNNode, v: BCPNNode) -> Tuple[bool, float]:
    """Granger因果显著性检验，带结果缓存"""
    cache_key = (u.node_id, v.node_id)
    if cache_key in _granger_cache:
        return _granger_cache[cache_key]
    
    try:
        test_data = np.column_stack([v.values, u.values])
        res = grangercausalitytests(test_data, maxlag=3, verbose=False)
        min_p = min([res[lag][0]['ssr_chi2test'][1] for lag in res.keys()])
        is_sig = min_p < config.SIG_THRESHOLD
        result = (is_sig, min_p)
    except Exception as e:
        logger.warning(f"Granger检验失败 [{u.node_id}→{v.node_id}]: {str(e)}")
        result = (False, 1.0)
    
    _granger_cache[cache_key] = result
    return result

def check_positive_monotonic(u: BCPNNode, v: BCPNNode) -> Tuple[bool, float, float]:
    """
    验证两个节点在异常时间段内是否呈现正向单调变化
    这是amplify传播的必要物理约束：源节点异常增加必须导致目标节点异常也增加
    返回：(是否满足正向单调, 相关系数, p值)
    """
    if not u.break_point:
        return True, 0.5, 1.0  # 源节点无异常，无法验证
    
    # 验证窗口：从源节点异常开始到观测结束
    start_idx = u.break_point
    u_values = u.values[start_idx:]
    v_values = v.values[start_idx:]
    
    if len(u_values) < 5 or len(v_values) < 5:
        return True, 0.5, 1.0  # 数据点不足，无法验证
    
    # 计算皮尔逊相关系数
    try:
        corr, p_val = pearsonr(u_values, v_values)
    except Exception as e:
        logger.warning(f"相关系数计算失败 [{u.node_id}→{v.node_id}]: {str(e)}")
        return True, 0.5, 1.0
    
    # 判断是否为显著的正向相关
    is_positive = corr > config.MIN_POSITIVE_CORR and p_val < config.CORR_SIG_THRESHOLD
    
    if not is_positive:
        ColorPrint.print_info(
            f"正向单调验证失败 [{u.node_id}→{v.node_id}]："
            f"相关系数={corr:.4f}, p值={p_val:.4f}"
        )
    else:
        print("正向单调验证成功")
    
    return is_positive, corr, p_val

# ==============================================
# 【✅ 终极修复：基于延迟后区间单调性的amplify时间一致性】
# ==============================================
def get_anomaly_interval(node: BCPNNode) -> Tuple[Optional[datetime], Optional[int], Optional[datetime], Optional[int]]:
    """
    增强版：返回异常区间的时间和索引
    返回：(开始时间, 开始索引, 结束时间, 结束索引)
    """
    if not node.is_anomaly or node.break_point is None:
        return None, None, None, None
    
    start_idx = node.break_point
    start_time = node.timestamps[start_idx]
    
    # 重新计算基线（和异常检测保持一致）
    baseline_len = int(len(node.values) * config.BASELINE_RATIO)
    baseline_data = node.values[:baseline_len]
    baseline_median = np.median(baseline_data)
    mad = np.median(np.abs(baseline_data - baseline_median))
    baseline_std = 1.4826 * mad if mad != 0 else 1e-6
    upper_threshold = baseline_median + config.MAD_THRESHOLD * baseline_std
    lower_threshold = baseline_median - config.MAD_THRESHOLD * baseline_std
    
    # 找到异常结束索引（连续3个点回到正常）
    end_idx = len(node.values) - 1
    for i in range(start_idx, len(node.values) - 2):
        if (lower_threshold <= node.values[i] <= upper_threshold and
            lower_threshold <= node.values[i+1] <= upper_threshold and
            lower_threshold <= node.values[i+2] <= upper_threshold):
            end_idx = i
            break
    
    end_time = node.timestamps[end_idx]
    
    ColorPrint.print_info(
        f"节点 [{node.node_id}] 异常区间："
        f"{start_time.strftime('%H:%M:%S')} → {end_time.strftime('%H:%M:%S')} "
        f"(持续 {(end_time - start_time).total_seconds():.1f} 秒，共 {end_idx - start_idx + 1} 个点)"
    )
    
    return start_time, start_idx, end_time, end_idx

def calculate_time_consistency(u: BCPNNode, v: BCPNNode, edge_type: str = "amplify") -> float:
    """
    【完全重写】严格区分两种传播的物理特性
    ✅ trigger：瞬时触发 → 单点时间差
    ✅ amplify：延迟后持续放大 → 搜索最优延迟τ，只看τ之后的区间单调性
    """
    u_start, u_start_idx, u_end, u_end_idx = get_anomaly_interval(u)
    v_start, v_start_idx, v_end, v_end_idx = get_anomaly_interval(v)
    
    if u_start is None or v_start is None:
        return 0.5
    
    # ===================== 绝对约束：因不能晚于果 =====================
    if v_start < u_start:
        ColorPrint.print_fail(
            f"时间一致性失败（{edge_type}）：[{u.node_id}] 晚于 [{v.node_id}]，因果倒置"
        )
        return 0.0
    
    # ===================== 分类型计算 =====================
    if edge_type == "trigger":
        # ---------------- trigger：保留原瞬时触发逻辑 ----------------
        delta_t = (v_start - u_start).total_seconds()
        
        if delta_t > config.MAX_TRIGGER_DELAY:
            return 0.3
        
        time_strength = np.exp(-((delta_t - config.OPTIMAL_TRIGGER_DELAY) ** 2) / (2 * (config.MAX_TRIGGER_DELAY / 3) ** 2))
        
        return round(max(0.0, time_strength), 4)
    
    else:  # amplify
        # ---------------- amplify：基于延迟后区间单调性的算法 ----------------
        ColorPrint.print_info(f"正在搜索 [{u.node_id}→{v.node_id}] 的最优传播延迟...")
        
        # 1. 提取两个节点的异常序列
        u_anomaly = u.values[u_start_idx:u_end_idx+1]
        v_anomaly = v.values[v_start_idx:v_end_idx+1]
        
        # 2. 计算可能的延迟范围（转换为采样点数）
        min_delay_points = int(config.MIN_AMPLIFY_DELAY / config.SAMPLING_INTERVAL)
        max_delay_points = int(config.MAX_AMPLIFY_DELAY / config.SAMPLING_INTERVAL)
        
        # 3. 搜索最优延迟τ，使得延迟后的u和v的相关性最高
        best_corr = -1.0
        best_tau = 0
        best_overlap_len = 0
        
        for tau in range(min_delay_points, min(max_delay_points, len(u_anomaly) - 5, len(v_anomaly) - 5)):
            # 对齐序列：u延迟τ个点后和v比较
            u_aligned = u_anomaly[tau:]
            v_aligned = v_anomaly[:len(u_aligned)]
            
            if len(u_aligned) < 5:
                continue
            
            # 计算皮尔逊相关系数
            try:
                corr, _ = pearsonr(u_aligned, v_aligned)
            except:
                continue
            
            # 记录最优相关
            if corr > best_corr:
                best_corr = corr
                best_tau = tau
                best_overlap_len = len(u_aligned)
        
        # 4. 如果没有找到有效相关，返回0
        if best_corr < config.MIN_POSITIVE_CORR:
            ColorPrint.print_info(
                f"amplify边无显著正相关：最优相关系数={best_corr:.4f} < {config.MIN_POSITIVE_CORR}"
            )
            return 0.0
        
        # 5. 计算最优延迟对应的实际时间
        best_tau_seconds = best_tau * config.SAMPLING_INTERVAL
        
        # 6. 计算覆盖度：有效重叠长度 / v的总异常长度
        coverage_ratio = best_overlap_len / len(v_anomaly) if len(v_anomaly) > 0 else 0.0
        
        # 7. 最终得分 = 相关性得分 * 覆盖度得分
        # 相关性得分：将[0.3, 1.0]映射到[0.0, 1.0]
        corr_score = (best_corr - config.MIN_POSITIVE_CORR) / (1.0 - config.MIN_POSITIVE_CORR)
        corr_score = max(0.0, min(1.0, corr_score))
        
        time_strength = corr_score * (0.7 + 0.3 * coverage_ratio)
        
        ColorPrint.print_success(
            f"✅ amplify边最优延迟：τ={best_tau_seconds:.1f}秒 | "
            f"相关系数={best_corr:.4f} | 覆盖度={coverage_ratio:.4f} | 时间一致性得分={time_strength:.4f}"
        )
        
        return round(max(0.0, time_strength), 4)

def check_temporal_consistency(u: BCPNNode, v: BCPNNode, edge_type: str) -> bool:
    """
    【适配新算法】时间一致性校验
    ✅ trigger：检查开始时间差
    ✅ amplify：只检查因果倒置，不检查任何固定延迟
    """
    u_start, _, _, _ = get_anomaly_interval(u)
    v_start, _, _, _ = get_anomaly_interval(v)
    
    if u_start is None or v_start is None:
        return True
    
    # 所有因果关系的绝对约束：因不能晚于果
    if v_start < u_start:
        ColorPrint.print_fail(
            f"时间一致性失败（{edge_type}）：[{u.node_id}] 晚于 [{v.node_id}]，因果倒置"
        )
        return False
    
    # 只有trigger边需要检查固定延迟范围
    if edge_type == "trigger":
        delta_t = (v_start - u_start).total_seconds()
        if delta_t < config.MIN_TRIGGER_DELAY or delta_t > config.MAX_TRIGGER_DELAY:
            ColorPrint.print_warn(
                f"trigger边延迟异常：{delta_t:.1f}秒 (正常范围：{config.MIN_TRIGGER_DELAY}-{config.MAX_TRIGGER_DELAY}秒)"
            )
    
    # amplify边不检查任何固定延迟，由算法自动搜索最优τ
    return True

def check_layer_consistency(u: BCPNNode, v: BCPNNode) -> bool:
    """校验定理1：层级一致性约束（仅用于trigger边）"""
    if u.depth >= v.depth:
        ColorPrint.print_fail(f"定理1失败：触发型仅允许自底向上 [{u.node_id}→{v.node_id}]")
        return False
    return True

# ==============================================
# 【LLM 语义一致性评分 优化版】
# ==============================================
def get_llm_semantic_score(u: BCPNNode, v: BCPNNode, edge_type: str) -> float:
    """
    严格对齐论文定义3.7：Ψ_LLM: V×V → [0,1]
    三大评估维度：层级约束合规性(33%)、物理逻辑合规性(33%)、领域知识合规性(33%)
    新增：所有传播类型都必须满足时间顺序约束
    """
    if not config.SILICONFLOW_API_KEY:
        ColorPrint.print_warn("未配置硅基流动API密钥，使用默认语义得分0.50")
        return 0.5
    
    try:
        system_prompt = """你是BCPN双向因果传播网络系统的**语义一致性评估模块**，必须严格遵守以下所有规则，禁止任何偏离。

# 一、BCPN四层层级精确定义（必须严格遵守）
BCPN将操作系统严格划分为四层，自底向上为：H→K→R→A，层级深度依次增加。

## 1. H层（Hardware Layer 硬件层）
- 核心职责：提供物理计算、存储、网络资源
- 典型实体：CPU核心、内存芯片、NVMe/SSD磁盘、物理网卡、电源模块、温度传感器、BMC
- 本质：所有软件运行的物理基础，其状态由硬件本身决定，不受上层软件直接修改

## 2. K层（Kernel Layer 内核层）
- 核心职责：硬件资源的抽象、调度与管理
- 典型实体：进程调度器、内存管理子系统、Block I/O栈、TCP/IP协议栈、文件系统、系统调用接口
- 本质：唯一具备直接访问硬件权限的软件层，向上层提供标准化的资源访问接口

## 3. R层（Runtime Layer 运行时层）
- 核心职责：为应用提供标准化运行环境与资源隔离
- 典型实体：容器运行时（Docker/containerd）、Kubernetes节点组件、JVM/Go Runtime、systemd守护进程、数据库进程、消息中间件
- 本质：运行在内核之上的软件基础设施，屏蔽底层硬件与内核差异

## 4. A层（Application Layer 应用层）
- 核心职责：实现具体业务逻辑
- 典型实体：微服务实例、Web服务器、API网关、前端应用、业务脚本
- 本质：面向用户的最上层软件，通过运行时层调用内核接口访问硬件资源

# 二、核心任务
给定两个系统节点A（源节点）和B（目标节点），以及边类型（触发型传播/放大型传播），评估"A的异常导致B的异常"这一因果关系是否符合操作系统的物理运行规律、层级约束与领域常识。

# 三、三大评估维度（权重均等，各占33%）
## 维度1：层级约束合规性（33%）【一票否决】
严格对齐BCPN两大核心定理，违反任意一条，本维度得0分，总分直接得0分。

### 定理1（触发型传播约束）
触发型传播（直接导致目标节点状态翻转）**只能自底向上**，即只能是：
- H→K、H→R、H→A
- K→R、K→A
- R→A
**绝对禁止**：K→H、R→H、R→K、A→H、A→K、A→R（触发型）

### 定理2（放大型传播约束）
放大型传播（仅影响目标节点资源指标，不直接导致其状态翻转）**可以双向**，包括：
- 同层级节点之间
- 上层节点→下层节点
- 下层节点→上层节点
**唯一限制**：上层节点的异常**永远无法独立触发下层节点的状态翻转**（即不能作为trigger边）

## 维度2：物理逻辑合规性（33%）
检查两个节点之间是否存在真实的系统依赖关系，传播方向是否符合资源流转方向：
✅ 合法依赖：下层为上层提供资源，上层依赖下层的服务；同层级组件存在直接交互
❌ 非法依赖：上层无法直接修改下层的物理状态或核心逻辑

### 常见合法物理依赖示例
- 磁盘（H）→ 内核IO子系统（K）：内核依赖磁盘提供存储服务
- 内核CPU调度器（K）→ 容器进程（R）：容器依赖内核分配CPU时间
- 数据库（R）→ 订单服务（A）：订单服务依赖数据库存储数据
- 订单服务（A）→ 数据库（R）：订单服务的请求会消耗数据库的CPU资源
- 应用锁等待（A）→ 内核锁等待（K）：应用层锁竞争会导致内核态锁等待增加

### 常见非法物理依赖示例
- 订单服务（A）→ 磁盘（H）：应用代码无法直接修改磁盘的物理介质
- 容器（R）→ 内核调度器（K）：容器无法直接修改内核的调度算法
- 应用（A）→ 内存芯片（H）：应用无法直接损坏物理内存

## 维度3：领域知识合规性（33%）
检查是否是运维领域公认的常见故障传播路径：
✅ 常见路径：行业内普遍认可的故障传播规律
❌ 罕见路径：没有明确证据支持的因果关系

### 常见合法故障传播示例
- 磁盘坏道（H）→ 内核IO延迟升高（K）
- 内核内存泄漏（K）→ 容器OOM（R）
- 数据库慢查询（R）→ 应用请求超时（A）
- 应用重试风暴（A）→ 数据库CPU打满（R）
- 网卡丢包（H）→ 应用网络超时（A）
- 应用文件锁竞争（A）→ 内核锁等待增加（K）

### 常见非法故障传播示例
- 应用代码bug（A）→ 磁盘硬件损坏（H）
- 容器重启（R）→ 内核panic（K）
- 前端页面卡顿（A）→ 服务器CPU升高（K）

# 四、绝对优先约束：时间顺序
无论其他条件如何，无论传播类型是trigger还是amplify，如果源节点的异常发生时间晚于目标节点的异常发生时间，这条因果关系的总分直接为0。
因果关系必须满足"因在前，果在后"的基本公理，没有任何例外。

# 五、评分标准
| 得分 | 含义 |
|------|------|
| 0.00 | 完全不可能，违反核心定理、物理规律或时间顺序 |
| 0.30 | 可能性极低，无明确的依赖关系 |
| 0.50 | 中性，无法确定是否存在因果关系 |
| 0.70 | 可能性较高，存在合理的依赖关系 |
| 0.90 | 几乎确定，是行业公认的标准故障传播路径 |
| 1.00 | 完全确定，存在直接的、不可替代的因果关系 |

# 六、输出要求（必须严格遵守）
1.  必须输出**严格的JSON格式**，禁止任何额外的解释、说明、markdown或自然语言
2.  必须包含以下字段，字段名不能修改：
    - `total_score`: 0-1之间的浮点数，精确到小数点后两位
    - `layer_score`: 层级约束合规性得分（0.00-0.33）
    - `layer_evidence`: 层级约束合规性的判断依据，必须引用BCPN定理
    - `physics_score`: 物理逻辑合规性得分（0.00-0.33）
    - `physics_evidence`: 物理逻辑合规性的判断依据
    - `domain_score`: 领域知识合规性得分（0.00-0.33）
    - `domain_evidence`: 领域知识合规性的判断依据
3.  所有判断依据必须严格基于输入的信息和上述BCPN规则，禁止任何无证据的推测
4.  如果输入信息不完整无法判断，输出：
    {"total_score": 0.50, "layer_score": 0.33, "layer_evidence": "层级约束合规", "physics_score": 0.00, "physics_evidence": "输入信息不足无法判断", "domain_score": 0.17, "domain_evidence": "输入信息不足无法判断"}

# 七、禁止事项
❌ 禁止参与根因推理或因果决策
❌ 禁止修改输入的节点信息或边类型
❌ 禁止添加任何输入中没有的信息
❌ 禁止输出任何非JSON格式的内容
❌ 禁止违反BCPN的任何核心定理
❌ 禁止忽略时间顺序约束"""

        user_prompt = f"""
节点A（源节点）：
  ID: {u.node_id}
  层级: {u.layer} ({config.LAYER_NAMES[u.layer]})
  异常时间: {u.break_time.strftime('%Y-%m-%d %H:%M:%S') if u.break_time else '未知'}
  异常描述: {u.node_description}

节点B（目标节点）：
  ID: {v.node_id}
  层级: {v.layer} ({config.LAYER_NAMES[v.layer]})
  异常时间: {v.break_time.strftime('%Y-%m-%d %H:%M:%S') if v.break_time else '未知'}
  异常描述: {v.node_description}

传播类型: {edge_type}

请输出语义一致性评估结果："""

        headers = {
            "Authorization": f"Bearer {config.SILICONFLOW_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"}
        }

        response = http_session.post(
            config.SILICONFLOW_API_URL, 
            headers=headers, 
            json=data, 
            timeout=config.LLM_TIMEOUT
        )
        response.raise_for_status()
        res_json = response.json()
        
        llm_output = json.loads(res_json["choices"][0]["message"]["content"].strip())
        
        required_fields = ["total_score", "layer_score", "physics_score", "domain_score"]
        if not all(field in llm_output for field in required_fields):
            raise ValueError("LLM输出缺少必要字段")
        
        total_score = float(llm_output["total_score"])
        total_score = max(0.0, min(1.0, total_score))
        
        ColorPrint.print_success(f"✅ LLM语义一致性评估完成 [{u.node_id}→{v.node_id}] ({edge_type})")
        print(f"  |- 总分: {total_score:.4f}")
        print(f"  |- 层级约束分: {float(llm_output['layer_score']):.4f} | 依据: {llm_output['layer_evidence']}")
        print(f"  |- 物理逻辑分: {float(llm_output['physics_score']):.4f} | 依据: {llm_output['physics_evidence']}")
        print(f"  |- 领域知识分: {float(llm_output['domain_score']):.4f} | 依据: {llm_output['domain_evidence']}")
        
        return total_score

    except Exception as e:
        ColorPrint.print_warn(f"⚠️ LLM调用或解析失败，使用默认得分0.50：{str(e)}")
        return 0.5

# ==============================================
# 【三元因果强度计算 严格对齐论文】
# ==============================================
class TriCausalStrengthCalculator:
    def calculate(self, u: BCPNNode, v: BCPNNode, causal_p: float, edge_type: str) -> float:
        """
        三元因果强度计算，严格对齐论文3.3.3节
        权重：统计因果(0.4) + 时间一致性(0.35) + 语义一致性(0.25)
        """
        ColorPrint.print_info(f"计算三元因果强度：[{u.node_id}] → [{v.node_id}] ({edge_type})")
        
        # 统计因果强度
        stat_strength = -np.log10(causal_p + 1e-10)
        stat_strength = max(0.0, min(1.0, stat_strength / 10.0))  # 归一化到[0,1]
        
        if stat_strength <= 0:
            return 0.0
        
        # 时间一致性（区分传播类型）
        time_strength = calculate_time_consistency(u, v, edge_type)
        
        # 因果倒置，直接返回0分
        if time_strength == 0.0:
            ColorPrint.print_info(f"因果倒置，过滤边 [{u.node_id}→{v.node_id}]")
            return 0.0
        
        # 语义一致性
        semantic_strength = get_llm_semantic_score(u, v, edge_type)
        
        # 加权求和
        final = (config.STAT_WEIGHT * stat_strength + 
                 config.TEMPORAL_WEIGHT * time_strength + 
                 config.SEMANTIC_WEIGHT * semantic_strength)
        
        return round(final, 4)

# ==============================================
# 【3.5.2 贝叶斯反向根因归因 严格对齐论文】
# ==============================================
def build_forward_propagation_matrix(nodes: List[BCPNNode], 
                                    edge_strength: Dict[Tuple[str, str], float],
                                    edge_type: Dict[Tuple[str, str], str]) -> np.ndarray:
    """
    构建正向传播概率矩阵P_f，严格对齐论文3.4.1节
    P_f[i,j]表示节点i的异常传播到节点j的概率
    """
    n = len(nodes)
    node_to_idx = {node.node_id: i for i, node in enumerate(nodes)}
    P_f = np.zeros((n, n), dtype=np.float64)
    
    for (u_id, v_id), strength in edge_strength.items():
        i = node_to_idx[u_id]
        j = node_to_idx[v_id]
        et = edge_type[(u_id, v_id)]
        
        if et == "trigger":
            P_f[i, j] = strength
        elif et == "amplify":
            P_f[i, j] = strength * config.AMPLIFICATION_DECAY
    
    # 自传播概率为1
    np.fill_diagonal(P_f, 1.0)
    
    return P_f

def calc_prior_prob(node: BCPNNode, all_nodes: List[BCPNNode]) -> float:
    """计算节点先验概率P(v_i)，基于历史故障频率与层级特性"""
    base_p = 0.1
    # 层级权重：深度越小(越底层)先验越高
    layer_weight = 1.0 / (1 + node.depth)
    # 固有缺陷加成
    defect_weight = 2.0 if node.has_inherent_defect else 1.0
    # 异常频次权重
    anomaly_cnt = sum(1 for n in all_nodes if n.is_anomaly)
    freq_weight = (1.0 + (1 if node.is_anomaly else 0)) / (1 + anomaly_cnt)
    
    prior = base_p * layer_weight * defect_weight * freq_weight
    return round(np.clip(prior, 0.01, 0.99), 4)

def calc_likelihood(root_node: BCPNNode, obs_set: List[BCPNNode], 
                    P_f: np.ndarray, node_to_idx: Dict[str, int]) -> float:
    """
    计算似然函数P(A_obs | v_i)，基于正向传播矩阵
    表示根因v_i引发所有观测异常节点的联合概率
    """
    if not root_node.is_anomaly:
        return 0.0
    
    root_idx = node_to_idx[root_node.node_id]
    total_prob = 1.0
    
    for obs_node in obs_set:
        if obs_node.node_id == root_node.node_id:
            continue
        obs_idx = node_to_idx[obs_node.node_id]
        trans_prob = P_f[root_idx, obs_idx]
        # 避免概率为0导致似然值全为0
        total_prob *= max(trans_prob, 1e-6)
    
    return round(total_prob, 6)

def bayes_root_cause_rank(nodes: List[BCPNNode], 
                         edge_strength: Dict[Tuple[str, str], float],
                         edge_type: Dict[Tuple[str, str], str]) -> List[BCPNNode]:
    """
    贝叶斯反向根因归因，严格对齐论文3.4.2节
    P(v_i | A_obs) ∝ P(A_obs | v_i) * P(v_i)
    """
    ColorPrint.print_info("开始执行【贝叶斯反向根因归因】(对齐论文3.5.2)")
    
    # 观测异常集合A_obs
    A_obs = [n for n in nodes if n.is_anomaly]
    if not A_obs:
        ColorPrint.print_warn("未检测到任何异常节点")
        return nodes
    
    # 构建正向传播矩阵
    P_f = build_forward_propagation_matrix(nodes, edge_strength, edge_type)
    node_to_idx = {node.node_id: i for i, node in enumerate(nodes)}
    
    # 计算每个节点的后验概率
    posteriors = []
    for node in nodes:
        prior = calc_prior_prob(node, nodes)
        likelihood = calc_likelihood(node, A_obs, P_f, node_to_idx)
        posterior = likelihood * prior
        
        node.prior_prob = prior
        node.likelihood = likelihood
        node.root_cause_score = posterior
        posteriors.append(posterior)
    
    # 归一化后验概率，提高区分度
    max_posterior = max(posteriors) if max(posteriors) > 0 else 1.0
    for node in nodes:
        node.root_cause_score = round(node.root_cause_score / max_posterior, 4)
    
    # 按后验概率降序排序
    ranked_nodes = sorted(nodes, key=lambda x: -x.root_cause_score)
    
    # 输出结果
    top_root = ranked_nodes[0]
    ColorPrint.print_success(f"🏆 贝叶斯最优根因节点：{top_root.node_id}")
    ColorPrint.print_info(f"根因后验概率得分（归一化）：{top_root.root_cause_score:.4f}")
    ColorPrint.print_info(f"原始先验概率：{top_root.prior_prob:.4f}，原始似然值：{top_root.likelihood:.6f}")
    
    return ranked_nodes

def check_identifiability_theorem(causal_graph: nx.DiGraph, anomaly_nodes: List[BCPNNode]) -> None:
    """
    校验定理4：根因可识别性定理
    在无环双向因果图中，若存在唯一入度为0的异常节点v*，且所有异常节点均存在从v*出发的可达路径，则根因唯一
    """
    ColorPrint.print_info("校验【定理4 根因可识别性定理】")
    
    if not anomaly_nodes:
        ColorPrint.print_warn("无异常节点，跳过定理校验")
        return
    
    # 检查图是否有环
    if not nx.is_directed_acyclic_graph(causal_graph):
        ColorPrint.print_warn("因果图存在环，不满足定理4的无环条件")
        return
    
    # 查找入度为0的异常节点
    in_degree_zero_anomaly = []
    for node in anomaly_nodes:
        in_deg = causal_graph.in_degree[node.node_id]
        if in_deg == 0:
            in_degree_zero_anomaly.append(node)
    
    if len(in_degree_zero_anomaly) != 1:
        ColorPrint.print_warn(f"不满足定理4：入度为0的异常节点共 {len(in_degree_zero_anomaly)} 个，根因不唯一")
        return
    
    v_star = in_degree_zero_anomaly[0]
    all_reachable = True
    
    # 检查所有异常节点是否都可达
    for obs_node in anomaly_nodes:
        if obs_node.node_id == v_star.node_id:
            continue
        if not nx.has_path(causal_graph, v_star.node_id, obs_node.node_id):
            all_reachable = False
            break
    
    if all_reachable:
        ColorPrint.print_success(f"✅ 定理4 成立：唯一根因 v* = {v_star.node_id}")
    else:
        ColorPrint.print_warn("❌ 定理4 不成立：存在异常节点无法从入度0节点可达")

# ==============================================
# 【BCPN 主模型 正向单调验证版】
# ==============================================
class BCPNModel:
    def __init__(self):
        self.nodes: List[BCPNNode] = []
        self.causal_graph = nx.DiGraph()
        self.edge_strength: Dict[Tuple[str, str], float] = {}
        self.edge_type: Dict[Tuple[str, str], str] = {}
        self.strength_calculator = TriCausalStrengthCalculator()
        self.self_excited_loops: List[List[str]] = []
        self.loop_gain: Dict[Tuple[str, ...], float] = {}
        self.scenario_info: Dict[str, Any] = {}
        self.logs: List[Dict[str, Any]] = []
        self.forward_propagation_matrix: Optional[np.ndarray] = None

    def fit(self, nodes: List[BCPNNode], scenario_info: Dict = None, logs: List[Dict] = None):
        """执行完整的BCPN根因分析流程"""
        self.nodes = ProductionDataLoader.align_timestamps(nodes)
        self.scenario_info = scenario_info or {}
        self.logs = logs or []
        
        # 清空缓存
        _granger_cache.clear()
        
        ColorPrint.print_step(1, "节点预处理：结构跃迁检测")
        self._preprocess_nodes()
        
        ColorPrint.print_step(2, "构建双向因果图")
        self._build_causal_graph()
       
        print("3333333333333333333333333333333333333333333333333333333333333")
        ColorPrint.print_step(3, "检测反馈环与自激级联故障")
        self._detect_self_excited_fault()
        
        ColorPrint.print_step(4, "贝叶斯反向根因归因")
        self._root_cause_inference()
        
        ColorPrint.print_step(5, "生成完整生产级分析报告")
        report = self.generate_production_report()
        print("\n" + report)
        
        # 保存报告到文件
        self._save_report(report)

    def _preprocess_nodes(self):
        """节点预处理：结构跃迁检测"""
        ColorPrint.print_info("开始检测所有节点的结构跃迁...")
        
        for node in tqdm(self.nodes, desc="检测结构跃迁"):
            if node.has_inherent_defect:
                ColorPrint.print_warn(f"节点 [{node.node_id}] 存在固有缺陷，故障风险较高")
            detect_structural_break(node)
        
        # 汇总异常节点信息
        anomaly_nodes = [n for n in self.nodes if n.is_anomaly]
        if anomaly_nodes:
            ColorPrint.print_success(f"\n===== 结构跃迁检测汇总 =====")
            ColorPrint.print_success(f"共检测到 {len(anomaly_nodes)} 个异常节点：")
            for node in sorted(anomaly_nodes, key=lambda x: x.break_time):
                print(f"  - {node.node_id:30} | "
                      f"时间：{node.break_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                      f"层级：{config.LAYER_NAMES[node.layer]}")
        else:
            ColorPrint.print_warn("未检测到任何异常节点")

    def _build_causal_graph(self):
        """
        构建双向因果图（正向单调验证版）
        严格对齐论文3.3.2节双类型传播定义
        新增：amplify边必须验证正向单调变化
        """
        for node in self.nodes:
            self.causal_graph.add_node(node.node_id)
        
        # 仅考虑异常节点之间的因果关系，大幅降低计算量
        anomaly_nodes = [n for n in self.nodes if n.is_anomaly]
        ColorPrint.print_info(f"检测到 {len(anomaly_nodes)} 个异常节点，仅在异常节点间构建因果图")
        
        edge_count = 0
        total_pairs = len(anomaly_nodes) * (len(anomaly_nodes) - 1)
        
        with tqdm(total=total_pairs, desc="构建因果边") as pbar:
            for u in anomaly_nodes:
                for v in anomaly_nodes:
                    if u.node_id == v.node_id:
                        pbar.update(1)
                        continue
                    
                    is_causal, p_val = causal_significance_test(u, v)
                    if not is_causal:
                        pbar.update(1)
                        continue
                    
                    # 双类型传播判断逻辑
                    edge_type = "none"
                    
                    # 触发型传播：必须同时满足4个条件
                    # 1. 自底向上传播（u.depth < v.depth）
                    # 2. 目标节点发生了状态翻转（v.state_flip=True）
                    # 3. 符合层级一致性约束（定理1）
                    # 4. 源节点异常严格早于目标节点异常（因果时间公理）
                    if u.depth < v.depth and v.state_flip:
                        if check_layer_consistency(u, v):
                            if check_temporal_consistency(u, v, "trigger"):
                                edge_type = "trigger"
                            else:
                                # 时间倒置的trigger边直接过滤
                                pbar.update(1)
                                continue
                    
                    # 放大型传播：所有其他情况
                    else:
                        if check_temporal_consistency(u, v, "amplify"):
                            edge_type = "amplify"
                        else:
                            # 时间倒置的amplify边直接过滤
                            pbar.update(1)
                            continue
                    
                    if edge_type == "none":
                        pbar.update(1)
                        continue
                    
                    # ===================== 核心新增：正向单调变化验证 =====================
                    # 对于amplify边，必须验证源节点异常后两者呈现正向单调变化
                    if edge_type == "amplify":
                        is_positive, corr, p_corr = check_positive_monotonic(u, v)
                        #print(u,v)
                        #print(is_positive)
                        if not is_positive:
                            ColorPrint.print_info(f"过滤非正向单调边 [{u.node_id}→{v.node_id}]")
                            pbar.update(1)
                            continue
                    # =====================================================================
                    
                    # 计算因果强度
                    strength = self.strength_calculator.calculate(u, v, p_val, edge_type)
                    
                    # 过滤弱边
                    if strength <= config.MIN_CAUSAL_STRENGTH:
                        ColorPrint.print_info(f"过滤弱边 [{u.node_id}→{v.node_id}]，强度：{strength:.4f}")
                        pbar.update(1)
                        continue
                    
                    # 添加边到因果图
                    self.causal_graph.add_edge(u.node_id, v.node_id, type=edge_type, strength=strength)
                    self.edge_strength[(u.node_id, v.node_id)] = strength
                    self.edge_type[(u.node_id, v.node_id)] = edge_type
                    
                    u.causal_out_edges.append((v.node_id, edge_type, strength))
                    v.causal_in_edges.append((u.node_id, edge_type, strength))
                    edge_count += 1
                    pbar.update(1)
        # 正反馈环补偿：处理长度为2的双向环
        self._compensate_feedback_loops()
        
        ColorPrint.print_success(f"因果图构建完成，共 {edge_count} 条有效边（过滤了 {total_pairs - edge_count} 条弱边/虚假边）")
        self._print_causal_graph_summary()

    def _compensate_feedback_loops(self):
        """正反馈环补偿机制：处理因采样间隔导致的双向边丢失"""
        ColorPrint.print_info("正在进行正反馈环补偿检测...")
        
        # 查找所有可能的双向环
        for u in self.nodes:
            if not u.is_anomaly:
                continue
            for v in self.nodes:
                if not v.is_anomaly or u.node_id == v.node_id:
                    continue
                
                # 如果已经存在u→v边，但不存在v→u边
                if (u.node_id, v.node_id) in self.edge_strength and (v.node_id, u.node_id) not in self.edge_strength:
                    u_node = u
                    v_node = v
                    
                    # 计算时间差
                    delta_t = (u_node.break_time - v_node.break_time).total_seconds()
                    
                    # 如果时间差小于采样间隔的2倍，认为是潜在的正反馈环
                    if abs(delta_t) < config.SAMPLING_INTERVAL * 2:
                        ColorPrint.print_info(
                            f"检测到潜在正反馈环：[{u.node_id}] ↔ [{v.node_id}]，时间差：{abs(delta_t):.1f}秒"
                        )
                        
                        # 计算反向边的强度
                        is_causal, p_val = causal_significance_test(v_node, u_node)
                        if is_causal:
                            # 反向边也需要验证正向单调变化
                            is_positive, _, _ = check_positive_monotonic(v_node, u_node)
                            if is_positive:
                                strength = self.strength_calculator.calculate(
                                    v_node, u_node, p_val, "amplify"
                                )
                                if strength > config.MIN_CAUSAL_STRENGTH:
                                    # 添加反向边，标记为"反馈环边"
                                    self.causal_graph.add_edge(v.node_id, u.node_id, type="amplify", strength=strength, is_feedback=True)
                                    self.edge_strength[(v.node_id, u.node_id)] = strength
                                    self.edge_type[(v.node_id, u.node_id)] = "amplify"
                                    ColorPrint.print_success(f"添加反馈环边：[{v.node_id}] → [{u.node_id}]，强度：{strength:.4f}")

    def _print_causal_graph_summary(self):
        """打印因果图汇总信息"""
        print("\n" + "="*110)
        print("【因果图汇总（仅显示有效边）】")
        print(f"{'源节点':30} {'目标节点':30} {'类型':8} {'强度':8} {'时间差(秒)':10} {'相关系数':10} {'备注':10}")
        for u, v, d in self.causal_graph.edges(data=True):
            u_node = next(n for n in self.nodes if n.node_id == u)
            v_node = next(n for n in self.nodes if n.node_id == v)
            delta_t = (v_node.break_time - u_node.break_time).total_seconds() if u_node.break_time and v_node.break_time else 0
            
            # 计算相关系数用于显示
            _, corr, _ = check_positive_monotonic(u_node, v_node)
            
            remark = "反馈环" if d.get("is_feedback", False) else ""
            print(f"{u:30} {v:30} {d['type']:8} {d['strength']:.4f} {delta_t:10.1f} {corr:10.4f} {remark:10}")
        print("="*110)

    def _detect_self_excited_fault(self):
        """检测反馈环与自激级联故障（优化版：只保留有意义的环）"""
        loops = list(nx.simple_cycles(self.causal_graph))
        ColorPrint.print_info(f"检测到 {len(loops)} 个原始反馈环")
        
        # 过滤掉增益过低的环
        meaningful_loops = []
        for loop in loops:
            gain = np.prod([self.edge_strength.get((loop[i], loop[(i+1)%len(loop)]), 0.0) 
                           for i in range(len(loop))])
            self.loop_gain[tuple(loop)] = gain
            
            if gain > config.MIN_LOOP_GAIN_TO_REPORT:
                meaningful_loops.append(loop)
                if gain > config.FEEDBACK_LOOP_CRITICAL_GAIN:
                    self.self_excited_loops.append(loop)
                    ColorPrint.print_warn(f"检测到自激级联故障环：{'→'.join(loop)}，增益：{gain:.4f}")
        
        ColorPrint.print_info(f"过滤后剩余 {len(meaningful_loops)} 个有意义的反馈环（增益>{config.MIN_LOOP_GAIN_TO_REPORT}）")

    def _root_cause_inference(self):
        """根因推理：贝叶斯反向归因 + 定理4校验"""
        # 贝叶斯反向根因排序
        self.nodes = bayes_root_cause_rank(self.nodes, self.edge_strength, self.edge_type)
        
        # 校验根因可识别性定理
        anomaly_nodes = [n for n in self.nodes if n.is_anomaly]
        check_identifiability_theorem(self.causal_graph, anomaly_nodes)

    def generate_production_report(self) -> str:
        """生成生产级根因分析报告"""
        report = "="*60 + "\nBCPN 生产级根因分析报告\n" + "="*60 + "\n"
        report += f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"场景名称：{self.scenario_info.get('scenario_name', '未知')}\n"
        report += f"场景ID：{self.scenario_info.get('scenario_id', '未知')}\n\n"
        
        anomaly_nodes = sorted([n for n in self.nodes if n.is_anomaly], 
                              key=lambda x: x.break_time if x.break_time else datetime.max)
        
        report += f"总节点数：{len(self.nodes)}\n异常节点数：{len(anomaly_nodes)}\n"
        report += f"有效因果边：{len(self.causal_graph.edges)}\n有意义反馈环：{len([g for g in self.loop_gain.values() if g > config.MIN_LOOP_GAIN_TO_REPORT])}\n"
        report += f"自激级联故障环：{len(self.self_excited_loops)}\n\n"
        
        # 异常节点状态跃迁时间线
        if anomaly_nodes:
            report += "【异常节点状态跃迁时间线】\n"
            report += f"{'节点ID':30} {'跃迁时间':20} {'层级':8} {'检测方法'}\n"
            report += "-"*80 + "\n"
            for node in anomaly_nodes:
                report += (f"{node.node_id:30} "
                          f"{node.break_time.strftime('%Y-%m-%d %H:%M:%S'):20} "
                          f"{config.LAYER_NAMES[node.layer]:8} "
                          f"{node.detection_method}\n")
            
            # 故障传播时间差
            if len(anomaly_nodes) > 1:
                report += "\n【故障传播时间差】\n"
                for i in range(1, len(anomaly_nodes)):
                    prev_node = anomaly_nodes[i-1]
                    curr_node = anomaly_nodes[i]
                    delta = (curr_node.break_time - prev_node.break_time).total_seconds()
                    report += f"{prev_node.node_id} → {curr_node.node_id}: {delta:.1f}秒\n"
        
        report += "\n【根因排序（贝叶斯后验概率，归一化）】\n"
        top3 = self.nodes[:3]
        for i, n in enumerate(top3):
            report += f"Top{i+1}：{n.node_id:30} | 后验得分：{n.root_cause_score:.4f} | 原始先验：{n.prior_prob:.4f} | 原始似然：{n.likelihood:.6f}\n"
            if n.is_anomaly and n.break_time:
                report += f"       状态跃迁时间：{n.break_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # 只显示有意义的反馈环
        meaningful_loops = [(loop, gain) for loop, gain in self.loop_gain.items() if gain > config.MIN_LOOP_GAIN_TO_REPORT]
        if meaningful_loops:
            report += "\n【有意义的反馈环（按增益排序）】\n"
            meaningful_loops.sort(key=lambda x: -x[1])
            for i, (loop, gain) in enumerate(meaningful_loops):
                report += f"环{i+1}：{'→'.join(loop)} | 增益：{gain:.4f}"
                if gain > config.FEEDBACK_LOOP_CRITICAL_GAIN:
                    report += " ⚠️ 自激级联故障"
                report += "\n"
        
        return report

    def _save_report(self, report: str):
        """保存报告到文件"""
        scenario_id = self.scenario_info.get("scenario_id", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"bcpn_report_{scenario_id}_{timestamp}.txt"
        
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report)
            ColorPrint.print_success(f"报告已保存到：{report_filename}")
        except Exception as e:
            ColorPrint.print_warn(f"保存报告失败：{str(e)}")

# ==============================================
# 【主程序 优化版】
# ==============================================
if __name__ == "__main__":
    np.random.seed(666)
    
    print("="*60)
    print("BCPN 生产级根因分析系统（正向单调验证版）")
    print("="*60)
    
    # 从环境变量获取API密钥
    config.SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
    #config.SILICONFLOW_API_KEY = "sk-jamzekcmgddqwnjrrekblfedqbgkpeixqnzatymeftfmgoxc"
    print(config.SILICONFLOW_API_KEY)
    if not config.SILICONFLOW_API_KEY:
        ColorPrint.print_warn("未设置SILICONFLOW_API_KEY环境变量，LLM语义评分将使用默认值0.5")
    
    # 命令行参数支持
    if len(sys.argv) > 1:
        scenario_json_path = sys.argv[1]
    else:
        scenario_json_path = "os_fault_dataset/A01_file_lock_contention/scenario_metadata.json"
    
    ColorPrint.print_info(f"使用数据路径：{scenario_json_path}")
    
    try:
        nodes, scenario_info, logs = ProductionDataLoader.load_from_multi_csv(scenario_json_path)
        if not nodes:
            ColorPrint.print_fail("未加载到任何有效节点，程序退出")
            sys.exit(1)
    except Exception as e:
        ColorPrint.print_fail(f"数据加载失败：{str(e)}")
        sys.exit(1)
    
    bcpn_model = BCPNModel()
    bcpn_model.fit(nodes, scenario_info, logs)
