from pydantic import BaseModel, HttpUrl
from typing import Optional,List

# Used when a user signs in (Google sends profile data)
class Doctor(BaseModel):
    id: Optional[int] = None  #optional because DB fills it
    google_id: str
    name: str
    picture: Optional[str] = None

    class Config:
        orm_mode = True

class AlertMessage(BaseModel):
    alerts: List[int]
