from pydantic import BaseModel, Field
from typing import Optional

class Transaction(BaseModel):
    tx_id: Optional[str] = None
    cc_num: Optional[str] = None
    trans_date_trans_time: Optional[str] = None
    amt: float
    category: Optional[str] = None
    merchant: Optional[str] = None
    gender: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[int] = None
    lat: Optional[float] = None
    long: Optional[float] = None
    merch_lat: Optional[float] = None
    merch_long: Optional[float] = None
    city_pop: Optional[int] = None
    hour: Optional[int] = None
    day: Optional[int] = None
    weekday: Optional[int] = None
    month: Optional[int] = None
