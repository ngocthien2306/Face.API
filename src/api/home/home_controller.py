from fastapi import APIRouter

router_root = APIRouter()


@router_root.get("/")
def read_root():
    return {"Hello": "World"}
