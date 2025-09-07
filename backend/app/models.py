from pydantic import BaseModel
from typing import Optional, List

# Used when a user signs in (Google sends profile data)
class Doctor(BaseModel):
    id: Optional[int] = None  # Optional because DB fills it
    google_id: str
    name: str
    picture: Optional[str] = None

    class Config:
        orm_mode = True


class AlertMessage(BaseModel):
    hours_until_sepsis: List[int]


class PatientSequence(BaseModel):
    features: list