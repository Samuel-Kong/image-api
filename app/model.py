# more imports!
# mainly for the datatype in the database
# also columns
from sqlalchemy import Column, String, Integer, DateTime, Text
from sqlalchemy import JSON as SA_JSON
from sqlalchemy.sql import func
# base class import from my other file
from .database import Base

# image table for each data required for the image
class Image(Base):
    # check table name
    __tablename__ = "images"

    # primary key
    id = Column(String, primary_key=True, index=True)
    # what it was named
    original_filename = Column(String, nullable=False)
    # where on earth is it stored
    original_path = Column(String, nullable=False)
    # thumbnails where!?
    thumbnail_small_path = Column(String, nullable=True)
    thumbnail_medium_path = Column(String, nullable=True)

    # image size
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    # what kinda format
    format = Column(String, nullable=True)
    size_bytes = Column(Integer, nullable=True)
    # remember exif and caption
    exif = Column(SA_JSON, nullable=True)
    caption = Column(Text, nullable=True)

    # stuff for the api
    status = Column(String, default="processing")  # this can be processing, done, failed
    error = Column(Text, nullable=True) # nullable since might not jave error

    # timing
    created_at = Column(DateTime(timezone=True), server_default=func.now()) # time of entry creation
    processed_at = Column(DateTime(timezone=True), nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
