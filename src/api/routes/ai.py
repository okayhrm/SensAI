from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import generate_draft from ai_service
from api.services.ai_service import generate_draft, review_draft, publish_to_sensai
from api.schemas.course_schemas import (
    AIResponse,
    AICourseDraft, # Keep this import
    AIReviewFeedback
    # Remove imports for CourseOutline, LessonDraft, MCQ, SAQ if they were only for hardcoding
)

router = APIRouter(prefix="/ai", tags=["ai-assistant"])

class TopicRequest(BaseModel):
    topic: str = Field(..., description="The topic for which to generate course content.")

@router.post("/generate-and-review", response_model=AIResponse)
async def generate_and_review_course(request: TopicRequest):
    """
    Generates a course draft, runs an AI review, and publishes it to SensAI.
    """
    try:
        # --- 1. Generate the course draft (Now using AI again) ---
        draft: AICourseDraft = await generate_draft(request.topic)
        # --- End AI Draft Generation ---

        # 2. Run the AI review pass
        review_feedback: AIReviewFeedback = await review_draft(draft)

        # 3. Publish to SensAI
        publish_status = await publish_to_sensai(draft)

        return AIResponse(
            draft=draft,
            review=review_feedback,
            publish_status=publish_status
        )
    except Exception as e:
        # Log the full traceback for any new errors here
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")