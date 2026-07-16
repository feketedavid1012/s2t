import datetime
import logging
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import (  # Import field_validator and ValidationInfo
    Field,
    ValidationInfo,
    computed_field,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()
LOGGER = logging.getLogger(__name__)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8', extra='ignore'
    )

    # GCP Config
    GENAI_GOOGLE_CLOUD_PROJECT: str = Field(..., alias='GENAI_GOOGLE_CLOUD_PROJECT')
    GENAI_GOOGLE_CLOUD_LOCATION: str = Field(..., alias='GENAI_GOOGLE_CLOUD_LOCATION')
    GENAI_TABLE_NAME: str = Field(..., alias='GENAI_TABLE_NAME')
    GENAI_BIGQUERY_PROJECT_ID: str = Field(..., alias='GENAI_BIGQUERY_PROJECT_ID')
    GENAI_BIGQUERY_API_ENDPOINT: str | None = Field(None, alias='GENAI_BIGQUERY_API_ENDPOINT')
    GENAI_MODEL_NAME: str = Field("gemini-2.0-flash", alias='GENAI_MODEL_NAME')

    # Other internal settings
    GENAI_LOG_LEVEL: str = Field(default='DEBUG', alias='GENAI_LOG_LEVEL')

    
    GENAI_SPANNER_EMB_COL: str = Field(..., alias='GENAI_SPANNER_EMB_COL')
    GENAI_SPANNER_MODEL_NAME: str = Field(..., alias='GENAI_SPANNER_MODEL_NAME')

    G_GENAI_BQ_FULL_ADDITIONAL_INFO_TABLE: str = Field(..., alias='GENAI_BQ_FULL_ADDITIONAL_INFO_TABLE')
    G_GENAI_SPANNER_INSTANCE: str = Field(..., alias='GENAI_SPANNER_INSTANCE')
    G_GENAI_SPANNER_DATABASE: str = Field(..., alias='GENAI_SPANNER_DATABASE')
    G_GENAI_SPANNER_TABLE_NAME: str = Field(..., alias='GENAI_SPANNER_TABLE_NAME')
    G_GENAI_DATASET_NAME: str = Field(..., alias='GENAI_DATASET_NAME')
    G_GENAI_TOP_K: int = Field(5, alias='GENAI_TOP_K')
    G_GENAI_RCA_TABLE_NAME: str = Field(..., alias='GENAI_RCA_TABLE_NAME')

    G_GENAI_EVAL_BQ_FULL_ADDITIONAL_INFO_TABLE: Optional[str] = Field(None, alias='GENAI_EVAL_BQ_FULL_ADDITIONAL_INFO_TABLE')
    G_GENAI_EVAL_SPANNER_INSTANCE: Optional[str] = Field(None, alias='GENAI_EVAL_SPANNER_INSTANCE')
    G_GENAI_EVAL_SPANNER_DATABASE: Optional[str] = Field(None, alias='GENAI_EVAL_SPANNER_DATABASE')
    G_GENAI_EVAL_SPANNER_TABLE_NAME: Optional[str] = Field(None, alias='GENAI_EVAL_SPANNER_TABLE_NAME')
    G_GENAI_EVAL_DATASET_NAME: Optional[str] = Field(None, alias='GENAI_EVAL_DATASET_NAME')
    G_GENAI_EVAL_TOP_K: Optional[int] = Field(5, alias='GENAI_EVAL_TOP_K')
    G_GENAI_EVAL_RCA_TABLE_NAME: Optional[str] = Field(None, alias='GENAI_EVAL_RCA_TABLE_NAME_CHAT')

    GENAI_DB_NAME: str = Field(..., alias='GENAI_DB_NAME')
    GENAI_SQL_INSTANCE: str = Field(..., alias='GENAI_SQL_INSTANCE')

    GENAI_CLOUD_SQL_DB_USERNAME: str = Field(..., alias='GENAI_CLOUD_SQL_DB_USERNAME')

    GENAI_MAX_QUERY_ROWS: int = Field(100, alias='GENAI_MAX_QUERY_ROWS')


    GENAI_IS_CLOUD: bool = Field(..., alias="GENAI_IS_CLOUD")
    GENAI_EVAL_ENV: bool = Field(False, alias="GENAI_EVAL_ENV")
    GENAI_UI_ENABLED: bool = Field(False, alias="GENAI_UI_ENABLED")

    GENAI_GEMINI_INPUT_TOKEN_PRICE: float = Field(0.1, alias='GENAI_GEMINI_INPUT_TOKEN_PRICE')
    GENAI_GEMINI_OUTPUT_TOKEN_PRICE: float = Field(0.4, alias='GENAI_GEMINI_OUTPUT_TOKEN_PRICE')
    GENAI_AVG_BYTES_PER_TOKEN: int = Field(4, alias='GENAI_AVG_BYTES_PER_TOKEN')
    GENAI_BQ_PRICE_PER_TB_USD: float = Field(6.00, alias='GENAI_BQ_PRICE_PER_TB_USD')
    GENAI_MAX_BYTES_BILLED: int = Field(100_000_000, alias='GENAI_MAX_BYTES_BILLED')
    GENAI_AVG_INPUT_TOKEN_OF_TICKETS : int = Field(1600, alias="GENAI_AVG_INPUT_TOKEN_OF_TICKETS")
    GENAI_AVG_OUTPUT_TOKEN_OF_TICKETS : int = Field(900, alias="GENAI_AVG_OUTPUT_TOKEN_OF_TICKETS")
    GENAI_TOKEN_USAGE_TABLE:str = Field(..., alias="GENAI_TOKEN_USAGE_TABLE")
    GENAI_DEFAULT_TOKEN_LIMIT : int = Field(1000000, alias="GENAI_DEFAULT_TOKEN_LIMIT")
    GENAI_PRICE_CALCULATION: bool = Field(False, alias="GENAI_PRICE_CALCULATION")
    
    GENAI_BYTES_IN_ONE_TB : int = 1_099_511_627_776
    GENAI_MILLION_TOKEN : int = 100_000_000

    @computed_field
    @property
    def DB_CONNECTION_STRING_FEEDBACK(self) -> str | None:
        if self.GENAI_IS_CLOUD:
            import urllib.parse

            instance_connection_name = f"{self.GENAI_GOOGLE_CLOUD_PROJECT}:{self.GENAI_GOOGLE_CLOUD_LOCATION}:{self.GENAI_SQL_INSTANCE}"
            encoded_username = urllib.parse.quote(self.GENAI_CLOUD_SQL_DB_USERNAME, safe='')

            url = f"postgresql+pg8000://{encoded_username}:@/{self.GENAI_DB_NAME}?unix_sock=/cloudsql/{instance_connection_name}/.s.PGSQL.5432"

            LOGGER.info(f"Connecting with IAM user: {self.GENAI_CLOUD_SQL_DB_USERNAME}")
            LOGGER.info(f"Socket path: /cloudsql/{instance_connection_name}/.s.PGSQL.5432")
            return url
        else:
            return
    
    
    @computed_field
    @property
    def SESSION_DB_URL(self) -> str | None:
        if self.GENAI_IS_CLOUD:
            import urllib.parse

            instance_connection_name = f"{self.GENAI_GOOGLE_CLOUD_PROJECT}:{self.GENAI_GOOGLE_CLOUD_LOCATION}:{self.GENAI_SQL_INSTANCE}"
            encoded_username = urllib.parse.quote(self.GENAI_CLOUD_SQL_DB_USERNAME, safe='')
            socket_path = f"/cloudsql/{instance_connection_name}"

            url = f"postgresql+asyncpg://{encoded_username}:@/{self.GENAI_DB_NAME}?host={socket_path}"

            LOGGER.info(f"Connecting with IAM user: {self.GENAI_CLOUD_SQL_DB_USERNAME}")
            LOGGER.info(f"Socket path: /cloudsql/{instance_connection_name}/.s.PGSQL.5432")
            return url
        else:
            return
        
    @computed_field
    @property
    def GENAI_CURRENT_TIME(self) -> datetime.datetime | None:
        if self.GENAI_EVAL_ENV:
            return datetime.datetime(2025, 6, 12, 0, 0)
        else:
            return datetime.datetime.now()
        
    @computed_field
    @property
    def GENAI_DATASET_NAME(self) -> str | None:
        if self.GENAI_EVAL_ENV:
            return self.G_GENAI_EVAL_DATASET_NAME
        else:
            return self.G_GENAI_DATASET_NAME
        
    @computed_field
    @property
    def GENAI_RCA_TABLE_NAME(self) -> str | None:
        if self.GENAI_EVAL_ENV:
            return self.G_GENAI_EVAL_RCA_TABLE_NAME
        else:
            return self.G_GENAI_RCA_TABLE_NAME
    
    @computed_field
    @property
    def GENAI_SPANNER_DATABASE(self) -> str | None:
        if self.GENAI_EVAL_ENV:
            return self.G_GENAI_EVAL_SPANNER_DATABASE
        else:
            return self.G_GENAI_SPANNER_DATABASE
    
    @computed_field
    @property
    def GENAI_SPANNER_TABLE_NAME(self) -> str | None:
        if self.GENAI_EVAL_ENV:
            return self.G_GENAI_EVAL_SPANNER_TABLE_NAME
        else:
            return self.G_GENAI_SPANNER_TABLE_NAME
    
    @computed_field
    @property
    def GENAI_BQ_FULL_ADDITIONAL_INFO_TABLE(self) -> str | None:
        if self.GENAI_EVAL_ENV:
            return self.G_GENAI_EVAL_BQ_FULL_ADDITIONAL_INFO_TABLE
        else:
            return self.G_GENAI_BQ_FULL_ADDITIONAL_INFO_TABLE
    
    @computed_field
    @property
    def GENAI_TOP_K(self) -> str | None:
        if self.GENAI_EVAL_ENV:
            return self.G_GENAI_EVAL_TOP_K
        else:
            return self.G_GENAI_TOP_K
    
    @computed_field
    @property
    def GENAI_SPANNER_INSTANCE(self) -> str | None:
        if self.GENAI_EVAL_ENV:
            return self.G_GENAI_EVAL_SPANNER_INSTANCE
        else:
            return self.G_GENAI_SPANNER_INSTANCE
    
    

    # --- CORRECTED V2 VALIDATOR ---
    @field_validator('GENAI_GOOGLE_CLOUD_PROJECT', 'GENAI_BIGQUERY_PROJECT_ID', mode='before')
    @classmethod # Validators are typically classmethods in V2
    def check_project_id(cls, v: Any, info: ValidationInfo) -> Any:
        # mode='before' corresponds to pre=True
        # always=True is often implied by checking if v is truthy,
        # or you can use check_fields=False if needed for specific scenarios
        field_name = info.field_name # Get the field name (or alias if used via Field)
        if not v:
            # Use info.field_name or simply rely on the error message context
            raise ValueError(f"'{field_name}' environment variable not set or empty.")
        if not isinstance(v, str): # Optional: Add type check if mode='before'
             raise ValueError(f"'{field_name}' must be a string.")
        return v
    # --- END CORRECTION ---

from functools import lru_cache


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
