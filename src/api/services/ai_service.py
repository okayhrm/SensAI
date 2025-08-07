import openai
from pydantic import ValidationError
import aiohttp
import json
import asyncio
from api.settings import settings
from api.schemas.course_schemas import (
    CourseOutline, 
    LessonDraft, 
    MCQ, 
    SAQ, 
    AICourseDraft, 
    AIReviewFeedback,
    AIResponse
)

def load_prompt(prompt_name: str) -> str:
    """Loads a prompt template from the prompts directory."""
    with open(f"{settings.ai_prompt_templates_path}/{prompt_name}", "r") as f:
        return f.read()

async def call_openai_with_schema(prompt: str, response_model, model_name: str):
    """
    Calls the OpenAI API with a prompt and validates the response against a Pydantic model.
    """
    try:
        client = openai.AsyncClient(base_url=settings.openai_base_url, api_key=settings.openai_api_key)
        response = await client.chat.completions.create(
            model=model_name,
            response_model=response_model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response
    except ValidationError as e:
        print(f"Validation Error: {e}")
        # Log the error and possibly return a default or raise a custom exception
        raise
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        raise

async def generate_draft(topic: str) -> AICourseDraft:
    """Generates the full course draft using multiple LLM calls."""
    outline_prompt = load_prompt("outline_prompt.txt").format(topic=topic)
    lesson_prompt = load_prompt("lesson_prompt.txt").format(topic=topic)
    questions_prompt = load_prompt("questions_prompt.txt").format(topic=topic)
    
    # Use aiohttp to run these concurrently for better performance
    outline_task = call_openai_with_schema(outline_prompt, CourseOutline, settings.ai_model_name)
    lesson_task = call_openai_with_schema(lesson_prompt, LessonDraft, settings.ai_model_name)
    questions_task = call_openai_with_schema(questions_prompt, list[MCQ | SAQ], settings.ai_model_name)
    
    outline, lesson, questions_raw = await asyncio.gather(outline_task, lesson_task, questions_task)
    
    # Simple parsing logic for the raw questions list
    mcqs = [q for q in questions_raw if isinstance(q, MCQ)]
    saqs = [q for q in questions_raw if isinstance(q, SAQ)]
    
    return AICourseDraft(outline=outline, lesson=lesson, mcqs=mcqs, saqs=saqs)

async def review_draft(draft: AICourseDraft) -> AIReviewFeedback:
    """Performs an AI and automated review pass on the generated draft."""
    content_for_review = json.dumps(draft.model_dump(), indent=2)
    review_prompt = load_prompt("reviewer_prompt.txt").format(content=content_for_review)

    # Call AI reviewer
    ai_review_feedback = await call_openai_with_schema(review_prompt, AIReviewFeedback, settings.ai_review_model_name)
    
    # Here you would integrate your other tools like the reading-level estimator
    # For now, these are just stubs
    reading_level_mismatch = await check_reading_level(draft.lesson.mini_project)
    duplicate_flags = await check_duplicates(draft)

    if reading_level_mismatch:
        ai_review_feedback.reading_level_mismatch = reading_level_mismatch
    if duplicate_flags:
        ai_review_feedback.duplicate_flags = duplicate_flags

    return ai_review_feedback

async def check_reading_level(text: str) -> str | None:
    """Stub for the reading-level estimator."""
    if settings.reading_level_api_url:
        # Example of making a request to an external service
        async with aiohttp.ClientSession() as session:
            async with session.post(settings.reading_level_api_url, json={"text": text}) as response:
                if response.status == 200:
                    result = await response.json()
                    # Implement logic to flag mismatches based on result
                    return result.get("mismatch_reason")
    return None

async def check_duplicates(draft: AICourseDraft) -> list[str]:
    """Stub for the duplicate detector."""
    if settings.duplicate_detector_api_url:
        # Example: check for duplicates in examples and questions
        all_text = [d.stem for d in draft.mcqs] + [d.stem for d in draft.saqs] + draft.lesson.examples
        async with aiohttp.ClientSession() as session:
            async with session.post(settings.duplicate_detector_api_url, json={"texts": all_text}) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("duplicate_flags", [])
    return []

async def publish_to_sensai(draft: AICourseDraft) -> dict:
    """Publishes the final draft to the SensAI API."""
    headers = {"Authorization": f"Bearer {settings.sensai_publish_api_key}"}
    payload = draft.model_dump()
    
    async with aiohttp.ClientSession() as session:
        async with session.post(settings.sensai_publish_api_url, json=payload, headers=headers) as response:
            return {"status_code": response.status, "body": await response.json()}