from pathlib import Path

from fastapi import Request
from fastapi.responses import FileResponse


async def create_no_cache_response(file_path: Path, request: Request) -> FileResponse:
    """创建禁止缓存的文件响应
    
    每次请求都会返回最新的文件内容，不使用浏览器缓存
    """
    response = FileResponse(file_path)
    
    # 添加禁止缓存的头信息
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    return response 