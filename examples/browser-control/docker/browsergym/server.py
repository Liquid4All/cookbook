"""
BrowserGym FastAPI服务器
提供本地MiniWoB任务API接口，兼容HF Space协议
"""
import os
import logging
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BrowserGym Local Server",
    description="本地BrowserGym环境API服务",
    version="1.0.0"
)

# 环境实例池（简化版，单实例）
env_instance: Optional[Any] = None
current_task: Optional[str] = None


class ResetRequest(BaseModel):
    task_name: str = "miniwob.click-test"
    observation_type: str = "ax_tree"
    headless: bool = True


class StepRequest(BaseModel):
    action: str


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "BrowserGym Local Server",
        "status": "running",
        "endpoints": ["/health", "/reset", "/step", "/close"]
    }


@app.get("/health")
async def health():
    """健康检查端点"""
    return {
        "status": "ok",
        "service": "browsergym-local",
        "env_initialized": env_instance is not None,
        "current_task": current_task
    }


@app.post("/reset")
async def reset(req: ResetRequest):
    """重置环境并开始新任务"""
    global env_instance, current_task
    
    try:
        logger.info(f"Resetting environment with task: {req.task_name}")
        
        # 延迟导入（避免启动时加载问题）
        from envs.browsergym_env import BrowserGymEnv
        
        # 如果已有实例，先关闭
        if env_instance is not None:
            try:
                env_instance.close()
            except Exception as e:
                logger.warning(f"Error closing previous env: {e}")
        
        # 创建新环境
        env_instance = BrowserGymEnv(
            task_name=req.task_name,
            observation_type=req.observation_type,
            headless=req.headless
        )
        current_task = req.task_name
        
        # 重置环境
        obs, info = env_instance.reset()
        
        logger.info(f"Environment reset successfully: {req.task_name}")
        
        return {
            "observation": str(obs),  # 转换为字符串避免序列化问题
            "info": info,
            "task_name": req.task_name
        }
        
    except Exception as e:
        logger.error(f"Reset error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def step(req: StepRequest):
    """执行一步动作"""
    global env_instance
    
    if env_instance is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    
    try:
        logger.info(f"Executing action: {req.action}")
        
        # 执行动作
        obs, reward, done, truncated, info = env_instance.step(req.action)
        
        logger.info(
            f"Step result - reward: {reward}, done: {done}, truncated: {truncated}"
        )
        
        return {
            "observation": str(obs),
            "reward": float(reward),
            "done": bool(done),
            "truncated": bool(truncated),
            "info": info
        }
        
    except Exception as e:
        logger.error(f"Step error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.post("/close")
async def close():
    """关闭环境"""
    global env_instance, current_task
    
    if env_instance is None:
        return {"status": "no active environment"}
    
    try:
        env_instance.close()
        env_instance = None
        task = current_task
        current_task = None
        
        logger.info(f"Environment closed: {task}")
        
        return {"status": "closed", "task": task}
        
    except Exception as e:
        logger.error(f"Close error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Close failed: {str(e)}")


@app.get("/status")
async def status():
    """获取服务状态"""
    return {
        "env_active": env_instance is not None,
        "current_task": current_task,
        "display": os.environ.get("DISPLAY", "not set")
    }


if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
