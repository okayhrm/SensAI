from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import enhanced functions from ai_service
from api.services.ai_service import generate_draft, review_draft, publish_to_sensai
from api.schemas.course_schemas import (
    AIResponse,
    AICourseDraft,
    AIReviewFeedback,
    TopicValidation, # Ensure these are correctly imported from schemas
    ContentBiasCheck # Ensure these are correctly imported from schemas
)

router = APIRouter(prefix="/ai", tags=["ai-assistant"])

class TopicRequest(BaseModel):
    topic: str = Field(..., description="The topic for which to generate course content.")

@router.post("/generate-and-review", response_model=AIResponse)
async def generate_and_review_course(request: TopicRequest):
    """
    Generates a course draft with validation, runs an AI review, and publishes it to SensAI.
    
    Enhanced with:
    - Topic validation for ambiguity and bias detection
    - Content bias checking
    - Comprehensive warnings system
    """
    try:
        # --- 1. Generate the course draft with enhanced validation ---
        # Unpack all four return values from generate_draft
        draft, topic_validation, bias_check, topic_warnings = await generate_draft(request.topic)
        
        # --- 2. Run the enhanced AI review pass ---
        # Pass topic_validation and bias_check to review_draft
        review_feedback = await review_draft(draft, topic_validation, bias_check)

        # --- 3. Check if safe to publish ---
        can_publish = True
        publish_blockers = []
        
        # Block publishing if severe bias detected
        if bias_check.has_bias and bias_check.severity_score >= 7:
            can_publish = False
            publish_blockers.append(f"High bias severity: {bias_check.severity_score}/10")
            
        # Block if quality too low (assuming review_feedback has this field)
        if review_feedback.overall_quality_score < 60:
            can_publish = False
            publish_blockers.append(f"Low quality score: {review_feedback.overall_quality_score}/100")
            
        # Block if topic not educational (from topic_validation)
        if not topic_validation.is_educational:
            can_publish = False
            publish_blockers.append("Topic not suitable for education")

        # --- 4. Publish to SensAI (only if safe) ---
        if can_publish:
            publish_status = await publish_to_sensai(draft)
        else:
            publish_status = {
                "status": "blocked",
                "message": f"Publishing blocked: {'; '.join(publish_blockers)}"
            }

        # --- 5. Collect all warnings for transparency ---
        warnings = []
        
        # Add warnings from topic validation
        warnings.extend(topic_warnings) # Add the warnings list from generate_draft
        
        # Add warnings from content review
        if review_feedback.reading_level_mismatch:
            warnings.append(f"Reading level issues: {review_feedback.reading_level_mismatch}")
            
        if review_feedback.duplicate_flags:
            warnings.append(f"Duplicates found: {len(review_feedback.duplicate_flags)} items")

        # Create the response, ensuring all Pydantic models are converted to dictionaries
        return AIResponse(
            draft=draft,
            review=review_feedback,
            publish_status=publish_status,
            topic_validation=topic_validation.model_dump(),
            bias_check=bias_check.model_dump(),
            warnings=warnings
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full traceback for any new errors here
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Optional: Add a topic validation endpoint for pre-checking
@router.post("/validate-topic")
async def validate_topic_only(request: TopicRequest):
    """
    Validate a topic without generating the full course.
    Useful for checking if a topic will work before full generation.
    """
    # Ensure validate_topic is imported from ai_service
    from api.services.ai_service import validate_topic
    
    try:
        validation = await validate_topic(request.topic)
        return {
            "topic": request.topic,
            "is_suitable": validation.is_educational and validation.is_specific,
            "validation_details": validation.model_dump(),
            "recommendation": (
                "Topic is suitable for course generation" 
                if validation.is_educational and validation.is_specific
                else f"Please clarify: {'; '.join(validation.suggested_clarifications)}"
            )
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Topic validation failed: {e}")