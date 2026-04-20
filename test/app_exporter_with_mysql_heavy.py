from prometheus_client import start_http_server, Counter, Gauge
import time
import random
import pymysql
from threading import Thread, Lock

# 应用层指标
REQUESTS = Counter('app_requests_total', 'Total requests')
RETRIES = Counter('app_retries_total', 'Total retries')
RESPONSE_TIME = Gauge('app_response_time_ms', 'Average response time')
DB_ERRORS = Counter('app_db_errors_total', 'Database errors')

# MySQL 连接配置
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'order_db',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
    'autocommit': False  # 关闭自动提交，增加事务开销
}

# 全局锁（用于线程安全）
lock = Lock()
# 预生成大量测试数据（内存中）
TEST_DATA = [(random.randint(1, 100000), 
              random.randint(1, 10000), 
              round(random.uniform(10, 10000), 2), 
              random.choice(['pending', 'paid', 'shipped', 'cancelled', 'refunded'])) 
             for _ in range(10000)]

def init_heavy_data():
    """【初始化】先插入100万条数据，让表变大，查询变慢"""
    print("正在初始化100万条测试数据（约需2分钟）...")
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            # 批量插入100万条
            for i in range(100):
                batch = TEST_DATA * 100  # 每次1万条
                cursor.executemany(
                    "INSERT INTO orders (user_id, product_id, amount, status) VALUES (%s, %s, %s, %s)",
                    batch
                )
                conn.commit()
                print(f"已插入 { (i+1)*10000 } 条数据")
    finally:
        conn.close()
    print("✅ 测试数据初始化完成！")

def heavy_db_task():
    """【重负载任务】复杂查询 + 批量更新 + 大事务"""
    conn = None
    try:
        conn = pymysql.connect(**DB_CONFIG)
        
        # 1. 复杂聚合查询（多表关联+排序+分组，非常消耗CPU）
        with conn.cursor() as cursor:
            sql = """
                SELECT 
                    user_id, 
                    COUNT(*) as order_count, 
                    SUM(amount) as total_amount,
                    AVG(amount) as avg_amount
                FROM orders 
                WHERE user_id BETWEEN %s AND %s
                GROUP BY user_id
                HAVING order_count > 5
                ORDER BY total_amount DESC
                LIMIT 100
            """
            cursor.execute(sql, (random.randint(1, 50000), random.randint(50001, 100000)))
            cursor.fetchall()
        
        # 2. 批量更新（大事务，消耗IO+CPU）
        with conn.cursor() as cursor:
            update_ids = [random.randint(1, 1000000) for _ in range(1000)]
            sql = "UPDATE orders SET status = 'updated', amount = amount * 1.01 WHERE id IN (%s)" % ','.join(['%s']*1000)
            cursor.execute(sql, update_ids)
        conn.commit()
        
        # 3. 随机读取（全表扫描概率）
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM orders ORDER BY RAND() LIMIT 10")
            cursor.fetchall()
            
    except Exception as e:
        with lock:
            DB_ERRORS.inc()
    finally:
        if conn:
            conn.close()

def worker_thread(thread_id):
    """工作线程：无限循环执行重负载任务"""
    print(f"线程 {thread_id} 启动")
    while True:
        start_time = time.time()
        
        # 执行重负载任务
        heavy_db_task()
        
        # 更新指标
        with lock:
            REQUESTS.inc()
            resp_time = (time.time() - start_time) * 1000
            RESPONSE_TIME.set(max(100, resp_time))
        
        # 【关键】几乎不 sleep，持续施压
        time.sleep(0.001)

if __name__ == '__main__':
    #start_http_server(8000)
    fault_start_time = time.time() + 300  # 5分钟后注入故障
    
    # 1. 先初始化100万条数据（仅第一次运行需要）
    # init_heavy_data()  # 注释掉这行如果已经初始化过
    
    # 2. 启动 50 个并发线程（大幅提升并发）
    print("启动 50 个工作线程...")
    for i in range(50):
        t = Thread(target=worker_thread, args=(i,), daemon=True)
        t.start()
    
    # 3. 主线程保持运行
    print("✅ 重负载应用启动成功，指标端口：8000")
    while True:
        time.sleep(1)
