# yay imports
# imports are for the engine, session and base class
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# i decided to use sqlite as it is simple to set up and doesnt require additional services
# can upgrade to either postgresql or others if needed
# use it to save the image data, exif and metadata
DATABASE_URL = "sqlite:///./images.db"

# create the database engine based on the url for the database
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
# config and create local ssession to handle the database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# my own function to initialise the databse
def init_db():
    Base.metadata.create_all(bind=engine)
