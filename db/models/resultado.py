from . import Base

from sqlalchemy import Column, Integer, ForeignKey


class Resultado(Base):
    __tablename__ = 'Clusterizacao__CIDADES_CONTRATOS'

    id = Column(Integer, ForeignKey('CIDADES_CONTRATOS.id'), primary_key=True)
    cluster = Column(Integer)
