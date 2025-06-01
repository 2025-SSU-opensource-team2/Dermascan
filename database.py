# database.py (수정)

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./sql_app.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    predicted_class = Column(String, index=True)
    scores_json = Column(String)
    image_path_raw = Column(String)
    image_path_encrypted = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# 새로 추가되는 DiseaseInfo 모델
class DiseaseInfo(Base):
    __tablename__ = "disease_info"

    id = Column(Integer, primary_key=True, index=True)
    disease_name = Column(String, unique=True, index=True, nullable=False) # 병명 (중복 불가, 필수)
    definition = Column(Text) # 정의
    symptoms = Column(Text) # 증상 (JSON 문자열 또는 콤마로 구분된 문자열)
    causes = Column(Text) # 원인
    treatment_methods = Column(Text) # 대처법/치료법
    recommended_medicine = Column(Text) # 추천 약
    precautions = Column(Text) # 주의사항/생활습관
    source_url = Column(String) # 출처 URL
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow) # 마지막 업데이트 시간

    __table_args__ = (UniqueConstraint('disease_name', name='_disease_name_uc'),) # 병명 유니크 제약 조건 명시

def init_db():
    Base.metadata.create_all(bind=engine)