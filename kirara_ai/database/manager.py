import os
from typing import Optional

from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from kirara_ai.ioc.container import DependencyContainer
from kirara_ai.logger import get_logger

logger = get_logger("DB")

# 创建Base类，用于所有ORM模型
Base = declarative_base()
metadata = MetaData()

class DatabaseManager:
    """数据库管理器，负责管理数据库连接和会话"""

    def __init__(self, container: DependencyContainer, database_url: Optional[str] = None, is_debug: bool = False):
        self.container = container
        self.engine = None
        self.session_factory = None
        self.data_dir = "./data/db"
        self.db_path = os.path.join(self.data_dir, "kirara.db")
        self.database_url = database_url
        self.is_debug = is_debug

    def initialize(self):
        """初始化数据库连接"""
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)

        # 创建数据库引擎
        if self.database_url:
            db_url = self.database_url
        else:
            db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(db_url, echo=self.is_debug)

        # 创建session工厂
        self.session_factory = sessionmaker(bind=self.engine)

        # 创建所有表
        Base.metadata.create_all(self.engine)

        logger.info(f"Database initialized at {self.engine.url}")

    def get_session(self) -> Session:
        """获取数据库会话"""
        if not self.session_factory:
            self.initialize()
        return self.session_factory()

    def shutdown(self):
        """关闭数据库连接"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
