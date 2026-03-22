import asyncio
import time
import httpx
import json
import statistics
from typing import List, Dict, Any

# 配置
BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{BASE_URL}/chat/message"
USER_ID = "benchmark_user"
SESSION_ID = "benchmark_session"

# 测试场景
SCENARIOS = [
    {
        "name": "简单问答 (QA)",
        "query": "什么是畅越套餐？"
    },
    {
        "name": "业务办理 (Handle)",
        "query": "我想办个网络电视。"
    },
    {
        "name": "账单查询 (Billing)",
        "query": "查询我的上个月账单。"
    }
]

async def measure_latency(query: str) -> Dict[str, Any]:
    """测量单次请求的延迟 (处理流式输出)"""
    start_time = time.perf_counter()
    first_token_time = None
    full_answer = ""
    
    payload = {
        "message": query,
        "user_id": USER_ID,
        "session_id": SESSION_ID
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", CHAT_ENDPOINT, json=payload) as response:
                if response.status_code != 200:
                    return {"success": False, "error": f"Status code: {response.status_code}", "duration": 0}
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data["type"] == "token":
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            full_answer += data["content"]
                        elif data["type"] == "done":
                            break
                        elif data["type"] == "error":
                            return {"success": False, "error": data["content"], "duration": 0}
                            
        end_time = time.perf_counter()
        duration = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else duration
        
        return {
            "success": True,
            "duration": duration,
            "ttft": ttft,
            "answer_length": len(full_answer)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "duration": 0
        }

async def run_benchmark(iterations: int = 3):
    print(f"开始进度性能测试，每个场景执行 {iterations} 次...")
    print("-" * 75)
    
    results = {}
    
    for scenario in SCENARIOS:
        name = scenario["name"]
        query = scenario["query"]
        print(f"正在测试场景: {name:<20} | Query: '{query}'")
        
        durations = []
        ttfts = []
        for i in range(iterations):
            res = await measure_latency(query)
            if res["success"]:
                durations.append(res["duration"])
                ttfts.append(res["ttft"])
                print(f"  迭代 {i+1}: 总耗时 {res['duration']:.2f}s | 首字延迟 (TTFT) {res['ttft']:.2f}s")
            else:
                print(f"  迭代 {i+1}: 失败 ({res['error']})")
                
        if durations:
            results[name] = {
                "avg": statistics.mean(durations),
                "ttft_avg": statistics.mean(ttfts),
                "max": max(durations)
            }
            
    print("\n" + "=" * 25 + " 性能测试摘要 " + "=" * 25)
    print(f"{'场景名称':<20} | {'平均总耗时':<12} | {'平均首字延迟':<12} | {'最大耗时':<10}")
    print("-" * 75)
    for name, stats in results.items():
        print(f"{name:<20} | {stats['avg']:>11.2f}s | {stats['ttft_avg']:>11.2f}s | {stats['max']:>9.2f}s")
    print("=" * 75)

if __name__ == "__main__":
    asyncio.run(run_benchmark())
