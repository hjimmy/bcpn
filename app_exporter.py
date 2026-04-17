# app_exporter.py
from prometheus_client import start_http_server, Counter, Gauge
import time
import random

# 应用层指标
REQUESTS = Counter('app_requests_total', 'Total requests')
RETRIES = Counter('app_retries_total', 'Total retries')
RESPONSE_TIME = Gauge('app_response_time_ms', 'Average response time')

def mock_order_service():
    """模拟订单服务，故障时会触发重试风暴"""
    REQUESTS.inc()
    # 正常响应时间100ms，故障时500ms以上
    if time.time() > fault_start_time:
        RESPONSE_TIME.set(random.normalvariate(600, 100))
        # 30%概率触发重试
        if random.random() < 0.8:
            RETRIES.inc()
            time.sleep(0.1)
    else:
        RESPONSE_TIME.set(random.normalvariate(100, 20))

if __name__ == '__main__':
    start_http_server(8000)
    fault_start_time = time.time() + 300  # 5分钟后注入故障
    print("应用启动成功，指标端口：8000")
    while True:
        mock_order_service()
        time.sleep(0.1)
