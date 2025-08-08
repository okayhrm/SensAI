import os
import tempfile
import random
from collections import defaultdict
import asyncio
from contextlib import asynccontextmanager
from fastapi import APIRouter, HTTPException, Body, BackgroundTasks, FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from typing import List, Optional, Dict, Literal, AsyncGenerator
import json
import instructor
import openai
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from os.path import exists
from api.config import openai_plan_to_model_name, UPLOAD_FOLDER_NAME
from api.models import (
    TaskAIResponseType,
    AIChatRequest,
    ChatResponseType,
    TaskType,
    GenerateCourseStructureRequest,
    GenerateCourseJobStatus,
    GenerateTaskJobStatus,
    QuestionType,
)
from api.llm import run_llm_with_instructor, stream_llm_with_instructor
from api.settings import settings
from api.utils.logging import logger # Assuming this is your custom logger
from api.utils.concurrency import async_batch_gather
from api.websockets import get_manager, router as websocket_router
from api.db.task import (
    get_task_metadata,
    get_question,
    get_task,
    get_scorecard,
    create_draft_task_for_course,
    store_task_generation_request,
    update_task_generation_job_status,
    get_course_task_generation_jobs_status,
    add_generated_learning_material,
    add_generated_quiz,
    get_all_pending_task_generation_jobs,
)
from api.db.course import (
    store_course_generation_request,
    get_course_generation_job_details,
    update_course_generation_job_status_and_details,
    update_course_generation_job_status,
    get_all_pending_course_structure_generation_jobs,
    add_milestone_to_course,
)
from api.db.chat import get_question_chat_history_for_user
from api.db.utils import construct_description_from_blocks
from api.utils.s3 import (
    download_file_from_s3_as_bytes,
    get_media_upload_s3_key_from_uuid,
)
from api.utils.audio import prepare_audio_input_for_ai
from api.settings import tracer
from opentelemetry.trace import StatusCode, Status
from openinference.instrumentation import using_attributes
from api.routes import (
    auth,
    code,
    cohort,
    course,
    org,
    task,
    chat,
    user,
    milestone,
    hva,
    file,
    # ai, # <-- REMOVED THIS SPECIFIC IMPORT, as we'll use the one below
    scorecard,
    sensai
)

# Import the ai router specifically from its module, allowing us to control its tag
from api.routes import ai as ai_router_module # Renamed to avoid conflict with `ai` variable below

from api.scheduler import scheduler
from bugsnag.asgi import BugsnagMiddleware
import bugsnag


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    os.makedirs(settings.local_upload_folder, exist_ok=True)
    # The resume functions were removed from ai.py, so these lines are commented out
    # asyncio.create_task(resume_pending_task_generation_jobs())
    # asyncio.create_task(resume_pending_course_structure_generation_jobs())
    yield
    scheduler.shutdown()

if settings.bugsnag_api_key:
    bugsnag.configure(
        api_key=settings.bugsnag_api_key,
        project_root=os.path.dirname(os.path.abspath(__file__)),
        release_stage=settings.env or "development",
        notify_release_stages=["development", "staging", "production"],
        auto_capture_sessions=True,
    )

app = FastAPI(lifespan=lifespan)

if settings.bugsnag_api_key:
    app.add_middleware(BugsnagMiddleware)

    @app.middleware("http")
    async def bugsnag_request_middleware(request: Request, call_next):
        bugsnag.configure_request(
            context=f"{request.method} {request.url.path}",
            request_data={
                "url": str(request.url),
                "method": request.method,
                "headers": dict(request.headers),
                "query_params": dict(request.query_params),
                "path_params": request.path_params,
                "client": {
                    "host": request.client.host if request.client else None,
                    "port": request.client.port if request.client else None,
                },
            },
        )
        response = await call_next(request)
        return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if exists(settings.local_upload_folder):
    app.mount(
        f"/{UPLOAD_FOLDER_NAME}",
        StaticFiles(directory=settings.local_upload_folder),
        name="uploads",
    )

app.include_router(file.router, prefix="/file", tags=["file"])
app.include_router(file.router, prefix="/file", tags=["file"])
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(task.router, prefix="/tasks", tags=["tasks"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(user.router, prefix="/users", tags=["users"])
app.include_router(org.router, prefix="/organizations", tags=["organizations"])
app.include_router(cohort.router, prefix="/cohorts", tags=["cohorts"])
app.include_router(course.router, prefix="/courses", tags=["courses"])
app.include_router(milestone.router, prefix="/milestones", tags=["milestones"])
app.include_router(scorecard.router, prefix="/scorecards", tags=["scorecards"])
app.include_router(code.router, prefix="/code", tags=["code"])
app.include_router(hva.router, prefix="/hva", tags=["hva"])
app.include_router(websocket_router, prefix="/ws", tags=["websockets"])
app.include_router(sensai.router, prefix="/sensai", tags=["sensai-publish"])
app.include_router(ai_router_module.router, tags=["ai-assistant"])
@app.get("/health")
async def health_check():
    return {"status": "ok"}