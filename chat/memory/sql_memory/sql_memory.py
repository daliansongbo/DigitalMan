import datetime
import jieba
import jieba.analyse


from sqlalchemy.orm import sessionmaker
from typing import List
from sqlalchemy import create_engine
from sqlalchemy import Column, String, Integer, Enum, DateTime 
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.ext.declarative import declarative_base
import enum 
from datetime import datetime
from sqlalchemy.orm import sessionmaker
import configs.config as config

import datetime

from pydantic import BaseModel
from sqlalchemy import Column, String, DateTime, Unicode
from sqlalchemy.inspection import inspect
from typing import Optional
from sqlalchemy import and_, or_
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()


class SHORT_MEMORY_SQL(Base):
    __tablename__ = "short_memory_history"
    id = Column(Integer, primary_key=True, index=True, nullable=False)
    user_name = Column(String(64), nullable=False)
    charater_name = Column(String(64), nullable=False)
    content = Column(Unicode(65535), nullable=True)
    tags =  Column(Unicode(65535), nullable=True)
    created_at = Column(DateTime(), nullable=False)
    updated_at = Column(DateTime(), nullable=False)
    
    def to_dict(self):
        return {
            c.key:
            getattr(self, c.key).isoformat() if isinstance(
                getattr(self, c.key), datetime.datetime) else getattr(
                    self, c.key)
            for c in inspect(self).mapper.column_attrs
        }

    def save(self, db):
        db.add(self)
        db.commit()


# TODO 搜索方式待整改
class SQLStorage():
    def __init__(self):
        self.engine = create_engine(config.DB_URI)
        self.Base = declarative_base(self.engine)
        self.session = sessionmaker(self.engine)()  # 构建session对象

    def search(self, charater_name: str,  user_name: str) -> List[str]:
        # 查询结果，并限制数量
        results = self.session.query(SHORT_MEMORY_SQL).filter(and_(SHORT_MEMORY_SQL.user_name==user_name,SHORT_MEMORY_SQL.charater_name==charater_name) ).order_by('-timestamp')[:config.SHORT_MEMORY_TOP_K]
        # 提取查询结果的 text 字段
        result_texts = [result.text for result in results]
        return result_texts

    def pageQuery(self, page_num: int, page_size: int, user_name: str,charater_name:str) -> List[str]:
        # 计算分页偏移量
        offset = (page_num - 1) * page_size
        # 分页查询，并提取 text 字段
        results = self.session.query(SHORT_MEMORY_SQL).filter(and_(SHORT_MEMORY_SQL.user_name==user_name,SHORT_MEMORY_SQL.charater_name==charater_name)).order_by('-timestamp').values_list('text', flat=True)[offset:offset + page_size]
        return list(results)

    def save(self, query_text: str, user_name: str,charater_name:str,query:str,responce:str) -> None:
        query_words = jieba.cut(query_text, cut_all=False)
        query_tags = list(query_words)
        keywords = jieba.analyse.extract_tags(" ".join(query_tags), topK=20)
        current_timestamp = datetime.datetime.now().isoformat()  #
        local_memory_model = SHORT_MEMORY_SQL(
            id = current_timestamp,
            user_name=user_name,
            charater_name = charater_name,
            content=query_text,
            tags=",".join(keywords),  # 设置标签
            created_at = current_timestamp,
            updated_at = current_timestamp
        )
        local_memory_model.save(self.session)




    def clear(self, owner: str) -> None:
        # 清除指定 owner 的记录
        self.session.query(SHORT_MEMORY_SQL).filter(owner=owner).delete()
