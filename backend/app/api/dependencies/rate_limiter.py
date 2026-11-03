import time
import uuid
from fastapi import HTTPException, Request, status
import redis.asyncio as aioredis
from app.core.config import settings

class RateLimiter:
    def __init__(self, max_request: int = 10, window_seconds: int = 60):
        self.max_request = max_request
        self.window_seconds = window_seconds
        self.redis: aioredis.Redis = aioredis.from_url(
            settings.redis_url,
            decode_responses = True
        )

    async def __call__(self, request: Request) -> None:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_id = forwarded_for.split(",")[0].strip()
        elif request.client:
            client_id = request.client.host
        else:
            client_id = "127.0.0.1"

        endpoint_path = request.url.path
        key = f"ratelimit:{endpoint_path}:{client_id}"

        now = time.time()
        clear_before = now - self.window_seconds

        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(key, 0, clear_before)
            pipe.zcard(key)
            pipe.zrange(key, 0, 0, withscores=True)
            _, current_count, oldest_entry = await pipe.execute()

        if current_count >= self.max_request:
            oldest_timestamp = oldest_entry[0][1] if oldest_entry else clear_before
            retry_after = max(1, int(oldest_timestamp + self.window_seconds - now))

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate Limit exceeded: Maximum {self.max_request} request per {self.window_seconds}",
                headers={"Retry-After": str(retry_after)}
            )

        async with self.redis.pipeline(transaction=True) as pipe:
            member = f"{now}:{uuid.uuid4().hex[:8]}"
            pipe.zadd(key, {member: now})
            pipe.expire(key, self.window_seconds)
            await pipe.execute()