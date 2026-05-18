from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import List
import uvicorn
from fastapi import Path
from fastapi.responses import JSONResponse

# ---------- FastAPI 应用实例 ----------
app = FastAPI(title="Agent Gateway", description="智能客服 Agent 系统", version="0.1.0")


# ---------- Pydantic 数据模型（对比 DRF Serializer） ----------
class UserOut(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool = True


# ---------- 模拟数据库（依赖注入用） ----------
fake_users_db = [
    {"id": 1, "username": "alice", "email": "alice@example.com", "is_active": True},
    {"id": 2, "username": "bob", "email": "bob@example.com", "is_active": False},
]


def get_db():
    # 模拟数据库会话，实际可以注入 SQLAlchemy 的 async session
    return fake_users_db


# ---------- 路由（对比 DRF ViewSet） ----------
@app.get("/", summary="根路径")
async def root():
    return {"message": "Agent Gateway 运行中"}


@app.get("/api/v1/users", response_model=List[UserOut], summary="用户列表")
async def list_users(
    active_only: bool = Query(False, description="是否只返回活跃用户"),
    db: list = Depends(get_db),  # 依赖注入，类似 DRF 的 get_queryset
):
    """获取用户列表，支持过滤活跃用户"""
    if active_only:
        users = [u for u in db if u["is_active"]]
    else:
        users = db
    return users


@app.get("/api/v1/users/{user_id}", response_model=UserOut, summary="用户详情")
async def get_user(
    user_id: int = Path(..., ge=1, description="用户ID"), db: list = Depends(get_db)
):
    """获取单个用户详情"""
    user = next((u for u in db if u["id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user


# ---------- 全局异常处理（加分项：统一错误格式） ----------
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.status_code, "message": exc.detail},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
