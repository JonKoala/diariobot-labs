from . import Base

from sqlalchemy import Column, Integer, String


class Diario_Classificacao(Base):
    __tablename__ = 'DIARIO_BOT_CLASSIFICACAO'

    id = Column(Integer, primary_key=True)
    corpo = Column(String)
    classe_id = Column(Integer)
