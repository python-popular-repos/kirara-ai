from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response


async def create_no_cache_response(file_path: Path, request: Request) -> Response:
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    stat = file_path.stat()
    mtime = stat.st_mtime_ns
    size = stat.st_size
    etag = f"{mtime}-{size}"

    if_none_match = request.headers.get("if-none-match")
    if if_none_match == etag:
        return Response(status_code=304)

    response = FileResponse(file_path)
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "no-cache"
    return response 