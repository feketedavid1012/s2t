
import importlib
import logging
import os
import sys
import sysconfig

import vertexai
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.cli.utils.agent_loader import AgentLoader
from google.cloud.logging_v2.handlers import StructuredLogHandler

from src.s2t_bench.config import get_settings

settings = get_settings()
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
os.environ["ADK_ENABLE_JSON_SCHEMA_FOR_FUNC_DECL"] = "1"


vertexai.init(
    project=settings.GENAI_GOOGLE_CLOUD_PROJECT,
    location=settings.GENAI_GOOGLE_CLOUD_LOCATION
)
level = settings.GENAI_LOG_LEVEL
h = StructuredLogHandler(stream=sys.stdout)
h.setLevel(logging.DEBUG)

logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', handlers=[h], force=True)
logging.captureWarnings(True)

for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "urllib3", "httpx"):
    lg = logging.getLogger(name)
    lg.handlers = []
    lg.propagate = True
    lg.setLevel(logging.NOTSET)

LOGGER = logging.getLogger(__name__)

importlib.reload(sysconfig)
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
agent_loader = AgentLoader(AGENT_DIR)



app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    session_service_uri=settings.SESSION_DB_URL,
    allow_origins=["*"],
    web=True,
)
