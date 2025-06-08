<<<<<<< HEAD
# database.py (수정)

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os # os 모듈 임포트 추가

# 로컬 개발 환경용 SQLite URL (필요 시 유지하거나, 제거해도 됨)
# DATABASE_URL = "sqlite:///./sql_app.db"

# Render 배포 환경용: 환경 변수에서 DATABASE_URL 가져오기
# Render 대시보드에서 설정할 DATABASE_URL 환경 변수를 참조합니다.
DATABASE_URL = os.getenv("DATABASE_URL")

# 만약 DATABASE_URL 환경 변수가 설정되지 않았다면 로컬 SQLite 사용 (개발용 폴백)
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./sql_app.db"
    print("Warning: DATABASE_URL environment variable not set, using local SQLite.")


# SQLAlchemy 엔진 생성
# SQLite는 여러 스레드에서 동시에 접근할 때 문제가 발생할 수 있으므로
# "check_same_thread": False 옵션을 추가합니다.
# PostgreSQL 등 다른 DB를 사용할 때는 connect_args={"check_same_thread": False} 제거해야 합니다.
# 여기서는 DATABASE_URL에 'sqlite'가 포함된 경우에만 이 옵션을 적용하도록 로직을 추가합니다.
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(DATABASE_URL)


# 데이터베이스 세션 클래스 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 모든 SQLAlchemy 모델의 기반이 되는 선언적 베이스 클래스
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

class DiseaseInfo(Base):
    __tablename__ = "disease_info"

    id = Column(Integer, primary_key=True, index=True)
    disease_name = Column(String, unique=True, index=True, nullable=False)
    definition = Column(Text)
    symptoms = Column(Text)
    causes = Column(Text)
    treatment_methods = Column(Text)
    recommended_medicine = Column(Text)
    precautions = Column(Text)
    source_url = Column(String)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (UniqueConstraint('disease_name', name='_disease_name_uc'),)

def init_db():
=======
# database.py (수정)

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os # os 모듈 임포트 추가

# 로컬 개발 환경용 SQLite URL (필요 시 유지하거나, 제거해도 됨)
# DATABASE_URL = "sqlite:///./sql_app.db"

# Render 배포 환경용: 환경 변수에서 DATABASE_URL 가져오기
# Render 대시보드에서 설정할 DATABASE_URL 환경 변수를 참조합니다.
DATABASE_URL = os.getenv("DATABASE_URL")

# 만약 DATABASE_URL 환경 변수가 설정되지 않았다면 로컬 SQLite 사용 (개발용 폴백)
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./sql_app.db"
    print("Warning: DATABASE_URL environment variable not set, using local SQLite.")


# SQLAlchemy 엔진 생성
# SQLite는 여러 스레드에서 동시에 접근할 때 문제가 발생할 수 있으므로
# "check_same_thread": False 옵션을 추가합니다.
# PostgreSQL 등 다른 DB를 사용할 때는 connect_args={"check_same_thread": False} 제거해야 합니다.
# 여기서는 DATABASE_URL에 'sqlite'가 포함된 경우에만 이 옵션을 적용하도록 로직을 추가합니다.
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(DATABASE_URL)


# 데이터베이스 세션 클래스 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 모든 SQLAlchemy 모델의 기반이 되는 선언적 베이스 클래스
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

class DiseaseInfo(Base):
    __tablename__ = "disease_info"

    id = Column(Integer, primary_key=True, index=True)
    disease_name = Column(String, unique=True, index=True, nullable=False)
    definition = Column(Text)
    symptoms = Column(Text)
    causes = Column(Text)
    treatment_methods = Column(Text)
    recommended_medicine = Column(Text)
    precautions = Column(Text)
    source_url = Column(String)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (UniqueConstraint('disease_name', name='_disease_name_uc'),)

def init_db():
>>>>>>> 675a081f317d4866ec04cc60a9eb59b6d169ad48
    Base.metadata.create_all(bind=engine)