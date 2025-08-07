from fastapi import APIRouter
from datetime import datetime

from api.schemas.course_schemas import AICourseDraft, PublishResponse

router = APIRouter(prefix="/sensai", tags=["sensai-publish"])

@router.post("/publish", response_model=PublishResponse)
async def publish_course_content(course_data: AICourseDraft):
    """
    Receives generated course content and simulates publishing it to the SensAI platform.
    """
    print(f"Received course for publishing: {course_data.outline.title}")

    return PublishResponse(
        message="Course content successfully published.",
        published_at=datetime.now(),
        course_title=course_data.outline.title
    )