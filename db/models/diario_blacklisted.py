from . import Base

from sqlalchemy import Column, Integer, String


class Diario_Backlisted(Base):
    __tablename__ = 'DIARIO_BOT_BLACKLISTED'

    id = Column(Integer, primary_key=True)
    palavra = Column(String)
