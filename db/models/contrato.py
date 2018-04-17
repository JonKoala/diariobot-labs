from . import Base

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship, synonym


class Contrato(Base):
    __tablename__ = 'CIDADES_CONTRATOS'

    id = Column(Integer, primary_key=True)
    ObjetoContrato = Column(String)
    ValorFirmado = Column(String)

    objeto = synonym('ObjetoContrato')
    valor = synonym('ValorFirmado')

    predicao = relationship('Predicao_Contrato', uselist=False)
