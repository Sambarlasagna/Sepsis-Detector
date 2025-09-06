from pydantic import BaseModel, HttpUrl
from typing import Optional

# Used when a user signs in (Google sends profile data)
class UserCreate(BaseModel):
    google_id: str
    name: str
    picture: Optional[HttpUrl] = None   # use HttpUrl to validate proper image link

# Used when returning user data from the DB
class UserResponse(BaseModel):
    id: int
    google_id: str
    name: str
    picture: Optional[str] = None
    class Config:
        orm_mode = True
