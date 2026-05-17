import numpy as np
import pandas as pd
from scipy.stats import chi2, f
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
from typing import List, Tuple, Dict, Optional
import json
import os
from datetime import datetime
import requests  # <-- 新增：用于调用硅基流动API

# ==============================================
# 【全局常量 严格对齐论文定义】
# ==============================================
LAYER_FULL_ORDER = {"H": 0, "K": 1, "R": 2, "A": 3}
LAYER_NAMES = {"H": "硬件层", "K": "内核层", "R": "运行时层", "A": "应用层"}
SIG_THRESHOLD = 0.05
PROB_THRESHOLD = 0.7
MIN_PROPAGATION_DELAY = 10
MAX_PROPAGATION_DELAY = 60
FEEDBACK_LOOP_CRITICAL_GAIN = 1.0
CHOW_SIG_THRESHOLD = 0.01

# ==============================================
# 【硅基流动 API 配置】
# ==============================================
SILICONFLOW_API_KEY = "sk-jamzekcmgddqwnjrrekblfedqbgkpeixqnzatymeftfmgoxc"  # 替换成你的硅基流动 KEY
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# ==============================================
# 【异常检测参数 工业级标准】
# ==============================================
BASELINE_RATIO = 0.3
MAD_THRESHOLD = 2.5
CONTINUOUS_ANOMALY_POINTS = 2

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
        
        self.is_anomaly: bool = False
        self.break_point: int = None
        self.break_time: datetime = None
        self.state_flip: bool = False
        self.detection_method: str = ""
        
        self.causal_in_edges: List[Tuple] = []
        self.causal_out_edges: List[Tuple] = []
        
        self.root_cause_score: float = 0.0

# ==============================================
# 【生产级数据加载】
# ==============================================
class ProductionDataLoader:
    @staticmethod
    def load_from_json(file_path: str) -> Tuple[List[BCPNNode], Dict, List[Dict]]:
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
        for node_data in data.get("nodes", []):
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
        for node_meta in nodes_meta:
            node_id = node_meta["node_id"]
            csv_filename = node_meta.get("csv_file", f"{node_id}.csv")
            csv_path = os.path.join(data_dir, csv_filename)
            
            if not os.path.exists(csv_path):
                ColorPrint.print_warn(f"节点 [{node_id}] 文件不存在，跳过")
                continue
            
            try:
                df = pd.read_csv(csv_path)
                time_col = "time"
                value_col = "value"
                
                timestamps = [pd.to_datetime(ts).to_pydatetime() for ts in df[time_col]]
                values = df[value_col].astype(float).values
                
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
        if not nodes:
            return nodes
        
        common_timestamps = set(nodes[0].timestamps)
        for node in nodes[1:]:
            common_timestamps.intersection_update(node.timestamps)
        
        common_timestamps = sorted(common_timestamps)
        ColorPrint.print_info(f"对齐时间戳：共 {len(common_timestamps)} 个公共时间点")
        
        aligned_nodes = []
        for node in nodes:
            ts_value_map = dict(zip(node.timestamps, node.values))
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
# 【工业级结构跃迁检测】
# ==============================================
def detect_structural_break(node: BCPNNode, p_thres: float = SIG_THRESHOLD) -> Tuple[bool, int]:
    ColorPrint.print_info(f"正在检测节点 [{node.node_id}] 的结构跃迁...")
    ts = node.values
    n = len(ts)
    if n < 20:
        ColorPrint.print_warn(f"节点时序长度不足")
        return False, None
    
    baseline_len = int(n * BASELINE_RATIO)
    baseline_data = ts[:baseline_len]
    baseline_median = np.median(baseline_data)
    mad = np.median(np.abs(baseline_data - baseline_median))
    baseline_std = 1.4826 * mad if mad != 0 else 1e-6
    upper_threshold = baseline_median + MAD_THRESHOLD * baseline_std
    lower_threshold = baseline_median - MAD_THRESHOLD * baseline_std
    
    over_threshold_mask = (ts > upper_threshold) | (ts < lower_threshold)
    sigma_break_point = None
    for t in range(baseline_len, n - CONTINUOUS_ANOMALY_POINTS + 1):
        if np.all(over_threshold_mask[t:t+CONTINUOUS_ANOMALY_POINTS]):
            sigma_break_point = t
            break
    
    sigma_break_valid = sigma_break_point is not None

    def improved_chow_test(y: np.ndarray, start_search: int, end_search: int) -> Tuple[Optional[int], float]:
        min_p = 1.0
        first_sig_break = None
        for t in range(start_search, end_search):
            n1, n2 = t, len(y) - t
            if n1 < 5 or n2 < 5:
                continue
            X = np.column_stack([np.ones(len(y)), np.arange(len(y))])
            beta_full = np.linalg.lstsq(X, y, rcond=None)[0]
            beta1 = np.linalg.lstsq(X[:t], y[:t], rcond=None)[0]
            beta2 = np.linalg.lstsq(X[t:], y[t:], rcond=None)[0]
            
            rss_full = np.sum((y - X @ beta_full) ** 2)
            rss1 = np.sum((y[:t] - X[:t] @ beta1) ** 2)
            rss2 = np.sum((y[t:] - X[t:] @ beta2) ** 2)
            rss_pooled = rss1 + rss2
            
            k = X.shape[1]
            f_stat = ((rss_full - rss_pooled) / k) / (rss_pooled / (len(y) - 2 * k))
            p_val = 1 - f.cdf(f_stat, k, len(y) - 2 * k)
            
            if p_val < min_p:
                min_p = p_val
            if p_val < CHOW_SIG_THRESHOLD:
                return t, p_val
        return None, min_p
    
    start_search = max(5, sigma_break_point - 10) if sigma_break_valid else baseline_len
    end_search = min(n-5, sigma_break_point + 10) if sigma_break_valid else n-5
    chow_break_point, min_chow_p = improved_chow_test(ts, start_search, end_search)
    chow_break_valid = chow_break_point is not None
    
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
        ColorPrint.print_success(f"节点 [{node.node_id}] 发生结构跃迁")
    
    return is_break, final_break_point

# ==============================================
# 【工具函数】
# ==============================================
def causal_significance_test(u: BCPNNode, v: BCPNNode) -> Tuple[bool, float]:
    try:
        test_data = np.column_stack([v.values, u.values])
        res = grangercausalitytests(test_data, maxlag=3, verbose=False)
        min_p = min([res[lag][0]['ssr_chi2test'][1] for lag in res.keys()])
        is_sig = min_p < SIG_THRESHOLD
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
        ColorPrint.print_fail(f"定理1失败：触发型仅允许自底向上")
        return False
    ColorPrint.print_success(f"定理1通过")
    return True

def check_bidirectional_boundary(u: BCPNNode, v: BCPNNode) -> bool:
    ColorPrint.print_info(f"校验【定理2：双向传播边界约束】")
    if u.depth > v.depth:
        if not v.has_inherent_defect and v.state_flip:
            ColorPrint.print_fail(f"定理2失败：下层无缺陷，上层无法触发故障")
            return False
        ColorPrint.print_success(f"定理2通过")
    return True

# ==============================================
# ================== 核心修改 ===================
# 【LLM 语义一致性评分：调用硅基流动 API】
# ==============================================
def get_llm_semantic_score(u: BCPNNode, v: BCPNNode, edge_type: str) -> float:
    """
    严格对齐论文定义3.7：Ψ_LLM: V×V → [0,1]
    三大评估维度：层级约束合规性(33%)、物理逻辑合规性(33%)、领域知识合规性(33%)
    输出严格JSON格式，包含总分、各维度得分和判断依据
    """
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
放大型传播（仅影响目标节点资源指标，不导致状态翻转）**可以双向**，但：
- 上层节点的异常**永远无法独立触发下层节点的状态翻转**
- 仅当下层节点本身存在固有缺陷时，上层异常才能加速其故障发生

## 维度2：物理逻辑合规性（33%）
检查两个节点之间是否存在真实的系统依赖关系，传播方向是否符合资源流转方向：
✅ 合法依赖：下层为上层提供资源，上层依赖下层的服务
❌ 非法依赖：上层无法直接修改下层的物理状态或核心逻辑

### 常见合法物理依赖示例
- 磁盘（H）→ 内核IO子系统（K）：内核依赖磁盘提供存储服务
- 内核CPU调度器（K）→ 容器进程（R）：容器依赖内核分配CPU时间
- 数据库（R）→ 订单服务（A）：订单服务依赖数据库存储数据
- 订单服务（A）→ 数据库（R）：订单服务的请求会消耗数据库的CPU资源

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

### 常见非法故障传播示例
- 应用代码bug（A）→ 磁盘硬件损坏（H）
- 容器重启（R）→ 内核panic（K）
- 前端页面卡顿（A）→ 服务器CPU升高（K）

# 四、评分标准
| 得分 | 含义 |
|------|------|
| 0.00 | 完全不可能，违反核心定理或物理规律 |
| 0.30 | 可能性极低，无明确的依赖关系 |
| 0.50 | 中性，无法确定是否存在因果关系 |
| 0.70 | 可能性较高，存在合理的依赖关系 |
| 0.90 | 几乎确定，是行业公认的标准故障传播路径 |
| 1.00 | 完全确定，存在直接的、不可替代的因果关系 |

# 五、输出要求（必须严格遵守）
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

# 六、禁止事项
❌ 禁止参与根因推理或因果决策
❌ 禁止修改输入的节点信息或边类型
❌ 禁止添加任何输入中没有的信息
❌ 禁止输出任何非JSON格式的内容
❌ 禁止违反BCPN的任何核心定理"""

        user_prompt = f"""
节点A（源节点）：
  ID: {u.node_id}
  层级: {u.layer} ({LAYER_NAMES[u.layer]})
  异常描述: {u.node_description}

节点B（目标节点）：
  ID: {v.node_id}
  层级: {v.layer} ({LAYER_NAMES[v.layer]})
  异常描述: {v.node_description}

传播类型: {edge_type}

请输出语义一致性评估结果："""

        headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"}
        }

        response = requests.post(SILICONFLOW_API_URL, headers=headers, json=data, timeout=20)
        response.raise_for_status()
        res_json = response.json()
        
        llm_output = json.loads(res_json["choices"][0]["message"]["content"].strip())
        
        required_fields = ["total_score", "layer_score", "physics_score", "domain_score"]
        if not all(field in llm_output for field in required_fields):
            raise ValueError("LLM输出缺少必要字段")
        
        total_score = float(llm_output["total_score"])
        total_score = max(0.0, min(1.0, total_score))
        
        ColorPrint.print_success(f"✅ LLM语义一致性评估完成")
        print(f"  |- 总分: {total_score:.4f}")
        print(f"  |- 层级约束分: {float(llm_output['layer_score']):.4f} | 依据: {llm_output['layer_evidence']}")
        print(f"  |- 物理逻辑分: {float(llm_output['physics_score']):.4f} | 依据: {llm_output['physics_evidence']}")
        print(f"  |- 领域知识分: {float(llm_output['domain_score']):.4f} | 依据: {llm_output['domain_evidence']}")
        
        return total_score

    except Exception as e:
        ColorPrint.print_warn(f"⚠️ LLM调用或解析失败，使用默认得分0.50：{str(e)}")
        return 0.5

        
# ==============================================
# 三元因果强度（已替换为真实LLM）
# ==============================================
class TriCausalStrengthCalculator:
    def calculate(self, u: BCPNNode, v: BCPNNode, causal_p: float, edge_type: str) -> float:
        ColorPrint.print_info(f"计算三元因果强度：[{u.node_id}] → [{v.node_id}]")
        stat_strength = -np.log10(causal_p + 1e-10)
        if stat_strength <= 0:
            return 0.0
        
        time_strength = calculate_time_consistency(u, v)
        semantic_strength = get_llm_semantic_score(u, v, edge_type)  # <-- 真实API调用
        
        final = 0.5 * stat_strength + 0.25 * time_strength + 0.25 * semantic_strength
        return round(final, 4)

# ==============================================
# BCPN 主模型
# ==============================================
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
            if node.has_inherent_defect:
                ColorPrint.print_warn(f"节点存在固有缺陷：{node.node_id}")
            is_break, break_point = detect_structural_break(node)
            node.is_anomaly = is_break
            node.break_point = break_point
            node.state_flip = is_break

    def _build_causal_graph(self):
        for node in self.nodes:
            self.causal_graph.add_node(node.node_id)
        
        edge_count = 0
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
                
                # 传入 edge_type 给 LLM
                strength = self.strength_calculator.calculate(u, v, p_val, edge_type)
                if strength <= 0: continue
                
                self.causal_graph.add_edge(u.node_id, v.node_id, type=edge_type, strength=strength)
                self.edge_strength[(u.node_id, v.node_id)] = strength
                self.edge_type[(u.node_id, v.node_id)] = edge_type
                
                u.causal_out_edges.append((v.node_id, edge_type, strength))
                v.causal_in_edges.append((u.node_id, edge_type, strength))
                edge_count += 1
        
        ColorPrint.print_success(f"因果图构建完成，共 {edge_count} 条边")
        self._print_causal_graph_summary()

    def _print_causal_graph_summary(self):
        print("\n" + "="*90)
        print("【因果图汇总】")
        print(f"{'源节点':30} {'目标节点':30} {'类型':8} {'强度':8}")
        for u, v, d in self.causal_graph.edges(data=True):
            print(f"{u:30} {v:30} {d['type']:8} {d['strength']:.4f}")
        print("="*90)

    def _detect_self_excited_fault(self):
        loops = list(nx.simple_cycles(self.causal_graph))
        for loop in loops:
            gain = np.prod([self.edge_strength.get((loop[i], loop[(i+1)%len(loop)]), 0.0) for i in range(len(loop))])
            self.loop_gain[tuple(loop)] = gain
            if gain > FEEDBACK_LOOP_CRITICAL_GAIN:
                self.self_excited_loops.append(loop)

    def _root_cause_inference(self):
        for node in self.nodes:
            if not node.is_anomaly:
                node.root_cause_score = 0.0
                continue
            in_trig = len([e for e in node.causal_in_edges if e[1] == "trigger"])
            base = 1.0 / (1 + in_trig)
            layer_w = 4.0 / (1 + node.depth)
            loop_s = sum(self.loop_gain.get(tuple(l), 0) * 0.3 for l in self.self_excited_loops if node.node_id in l)
            node.root_cause_score = round(base * 0.5 + layer_w * 0.3 + loop_s * 0.2, 4)
        
        self.nodes.sort(key=lambda x: -x.root_cause_score)
        ColorPrint.print_success(f"🏆 Top1 根因：{self.nodes[0].node_id}")

    def generate_production_report(self) -> str:
        report = "="*60 + "\nBCPN 生产级根因分析报告\n" + "="*60 + "\n"
        anomaly_nodes = sorted([n for n in self.nodes if n.is_anomaly], key=lambda x: x.break_point)
        report += f"总节点数：{len(self.nodes)}\n异常节点数：{len(anomaly_nodes)}\n"
        report += f"因果边：{len(self.causal_graph.edges)}\n反馈环：{len(self.loop_gain)}\n\n"
        top3 = self.nodes[:3]
        for i, n in enumerate(top3):
            report += f"Top{i+1}：{n.node_id} | 得分：{n.root_cause_score:.4f}\n"
        return report

# ==============================================
# 主程序
# ==============================================
if __name__ == "__main__":
    np.random.seed(666)
    
    print("="*60)
    print("BCPN 生产级根因分析系统（硅基流动 LLM 版）")
    print("="*60)
    
    scenario_json_path = "os_fault_dataset/A01_file_lock_contention/scenario_metadata.json"
    
    try:
        nodes, scenario_info, logs = ProductionDataLoader.load_from_multi_csv(scenario_json_path)
        if not nodes:
            exit(1)
    except:
        exit(1)
    
    bcpn_model = BCPNModel()
    bcpn_model.fit(nodes, scenario_info, logs)

