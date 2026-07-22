import os
import json
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()

class Student(Base):
    __tablename__ = 'students'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    # Store the 128D signature as a JSON string
    signature = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    attendances = relationship("AttendanceRecord", back_populates="student", cascade="all, delete-orphan")
    
    def get_signature(self):
        if self.signature:
            return json.loads(self.signature)
        return None

    def set_signature(self, signature_list):
        self.signature = json.dumps(signature_list)

class AttendanceRecord(Base):
    __tablename__ = 'attendance'
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey('students.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    date_str = Column(String, index=True, nullable=False) # For easy daily querying
    confidence = Column(Float, nullable=False)
    
    student = relationship("Student", back_populates="attendances")

# Create sqlite database
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'attendance_v2.db')
engine = create_engine(f'sqlite:///{DB_PATH}', connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
