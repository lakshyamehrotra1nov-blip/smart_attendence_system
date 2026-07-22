from sqlalchemy.orm import Session
from datetime import datetime
from database.models import Student, AttendanceRecord
import json

def get_student_by_name(db: Session, name: str):
    return db.query(Student).filter(Student.name == name).first()

def get_all_students(db: Session):
    return db.query(Student).all()

def register_student(db: Session, name: str, signature: list):
    """Registers a new student with their signature."""
    # Check if student exists
    student = db.query(Student).filter(Student.name == name).first()
    if not student:
        student = Student(name=name)
        db.add(student)
        
    student.set_signature(signature)
    db.commit()
    db.refresh(student)
    return student

def log_attendance(db: Session, student_id: int, confidence: float):
    """Logs attendance if not already marked today."""
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    
    # Check if already present today
    existing = db.query(AttendanceRecord).filter(
        AttendanceRecord.student_id == student_id,
        AttendanceRecord.date_str == date_str
    ).first()
    
    if existing:
        return False, existing # Already marked
        
    record = AttendanceRecord(
        student_id=student_id,
        timestamp=now,
        date_str=date_str,
        confidence=confidence
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return True, record

def get_today_attendance(db: Session):
    date_str = datetime.now().strftime('%Y-%m-%d')
    records = db.query(AttendanceRecord).filter(AttendanceRecord.date_str == date_str).all()
    return records
