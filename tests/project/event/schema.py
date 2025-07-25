from pydantic import BaseModel


class PingRequest(BaseModel):
    p1: int
    p2: int


class BaseResponse(BaseModel):
    status: str = "ok"
    message: str = "ok"


class PingRequest2(BaseModel):
    p2_1: int
    p2_2: int
    p2_3: int
