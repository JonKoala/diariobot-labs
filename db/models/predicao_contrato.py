from . import Base

from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.orm import relationship


class Predicao_Contrato(Base):
    __tablename__ = 'Predicao_CIDADES_CONTRATOS'

    id = Column(Integer, ForeignKey('CIDADES_CONTRATOS.id'), primary_key=True)
    classe = Column(Integer)
    cluster = Column(Integer)

    contrato = relationship('Contrato', uselist=False)
