# load_disease_data.py
import json
from sqlalchemy.orm import Session
from database import SessionLocal, init_db, DiseaseInfo, Prediction # Prediction도 포함 (init_db에 필요)

def load_data_to_db():
    init_db() # 테이블이 없으면 생성

    db: Session = SessionLocal()
    try:
        with open("skin_disease_data.json", "r", encoding="utf-8") as f:
            diseases_data = json.load(f)

        for data in diseases_data:
            disease_name = data.get("병명")
            if not disease_name:
                print(f"경고: 병명 정보가 없는 항목이 있습니다. 스킵합니다: {data}")
                continue

            # 이미 존재하는 병명인지 확인
            existing_disease = db.query(DiseaseInfo).filter(DiseaseInfo.disease_name == disease_name).first()

            if existing_disease:
                # 이미 존재하면 업데이트 (필요 시)
                # 여기서는 단순히 정의만 업데이트하는 예시. 필요에 따라 다른 필드도 업데이트 가능
                existing_disease.definition = data.get("정의", "")
                existing_disease.symptoms = json.dumps(data.get("증상", [])) # 리스트를 JSON 문자열로 저장
                existing_disease.causes = json.dumps(data.get("원인", []))
                existing_disease.treatment_methods = json.dumps(data.get("대처법/치료법", []))
                existing_disease.recommended_medicine = json.dumps(data.get("추천_약", []))
                existing_disease.precautions = json.dumps(data.get("주의사항/생활습관", []))
                existing_disease.source_url = data.get("출처", "")
                print(f"'{disease_name}' 정보 업데이트 완료.")
            else:
                # 존재하지 않으면 새로 추가
                new_disease = DiseaseInfo(
                    disease_name=disease_name,
                    definition=data.get("정의", ""),
                    symptoms=json.dumps(data.get("증상", [])),
                    causes=json.dumps(data.get("원인", [])),
                    treatment_methods=json.dumps(data.get("대처법/치료법", [])),
                    recommended_medicine=json.dumps(data.get("추천_약", [])),
                    precautions=json.dumps(data.get("주의사항/생활습관", [])),
                    source_url=data.get("출처", "")
                )
                db.add(new_disease)
                print(f"'{disease_name}' 정보 추가 완료.")
        
        db.commit()
        print("모든 병명 데이터베이스 저장/업데이트 완료.")

    except FileNotFoundError:
        print("오류: 'skin_disease_data.json' 파일을 찾을 수 없습니다. 크롤링을 먼저 실행해주세요.")
    except json.JSONDecodeError:
        print("오류: 'skin_disease_data.json' 파일이 유효한 JSON 형식이 아닙니다.")
    except Exception as e:
        db.rollback() # 오류 발생 시 롤백
        print(f"데이터베이스 저장 중 오류 발생: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    load_data_to_db()