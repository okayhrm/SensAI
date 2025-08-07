import openai
from pydantic import ValidationError, BaseModel
import aiohttp
import json
import asyncio
from typing import List, Dict, Any, TypeVar, Type, Union
import os
from fastapi import HTTPException

from api.settings import settings
from api.schemas.course_schemas import (
    CourseOutline,
    LessonDraft,
    MCQ,
    SAQ,
    AICourseDraft,
    AIReviewFeedback,
    PublishResponse
)
from api.utils.logging import logger

PydanticModel = TypeVar('PydanticModel', bound=BaseModel)
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1

def load_prompt(prompt_name: str) -> str:
    """Loads a prompt template from the prompts directory."""
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(os.path.dirname(current_file_dir))
        prompt_full_path = os.path.join(src_dir, settings.ai_prompt_templates_path, prompt_name)

        with open(prompt_full_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {prompt_full_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading prompt {prompt_name}: {e}")
        raise

async def call_openai_structured(
    prompt: str,
    target_model: Type[PydanticModel],
    model_name: str
) -> PydanticModel:
    """
    Calls OpenAI API and validates against Pydantic model.
    Falls back to simple approach if structured output fails.
    """
    client = openai.AsyncClient(base_url=settings.openai_base_url, api_key=settings.openai_api_key)

    for attempt in range(MAX_RETRIES):
        try:
            # Try with response_format first, fallback if not supported
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a course creation assistant. Always respond with valid JSON only. No explanations, no markdown, just pure JSON."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    response_format={"type": "json_object"}
                )
                logger.info(f"DEBUG: Using structured output mode")
            except Exception as structured_error:
                logger.warning(f"Structured output not supported, falling back: {structured_error}")
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a course creation assistant. Always respond with valid JSON only. No explanations, no markdown, just pure JSON."
                        },
                        {
                            "role": "user", 
                            "content": prompt + "\n\nIMPORTANT: Return ONLY valid JSON, no other text."
                        }
                    ]
                )
                logger.info(f"DEBUG: Using fallback mode")
            
            raw_content = response.choices[0].message.content.strip()
            logger.info(f"DEBUG (Attempt {attempt+1}): Raw response length: {len(raw_content)}")
            logger.info(f"DEBUG: First 300 chars: {raw_content[:300]}")
            
            # Try to parse JSON directly
            try:
                parsed_json = json.loads(raw_content)
                logger.info(f"DEBUG: Successfully parsed JSON directly")
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from markdown or other formatting
                import re
                json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw_content, re.DOTALL)
                if json_match:
                    cleaned_content = json_match.group(1).strip()
                    logger.info(f"DEBUG: Extracted JSON from code block")
                else:
                    # Try to find JSON object boundaries
                    json_match = re.search(r'(\{.*\})', raw_content, re.DOTALL)
                    if json_match:
                        cleaned_content = json_match.group(1).strip()
                        logger.info(f"DEBUG: Extracted JSON from object boundaries")
                    else:
                        raise json.JSONDecodeError("No valid JSON found", raw_content, 0)
                
                parsed_json = json.loads(cleaned_content)
                logger.info(f"DEBUG: Successfully parsed JSON after extraction")
            
            # Debug the parsed structure
            logger.info(f"DEBUG: Parsed JSON keys: {list(parsed_json.keys()) if isinstance(parsed_json, dict) else 'Not a dict'}")
            
            # Validate with Pydantic
            validated_model = target_model(**parsed_json)
            logger.info(f"DEBUG: Successfully validated against {target_model.__name__}")
            return validated_model

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error (Attempt {attempt+1}): {e}")
            logger.error(f"Raw content: {raw_content}")
        except ValidationError as e:
            logger.error(f"Pydantic validation error (Attempt {attempt+1}): {e}")
            logger.error(f"Validation errors: {e.errors()}")
            if 'parsed_json' in locals():
                logger.error(f"Parsed JSON structure: {json.dumps(parsed_json, indent=2)}")
        except openai.APIStatusError as e:
            logger.error(f"OpenAI API error (Attempt {attempt+1}): {e}")
            logger.error(f"API error details: {e.response.text if hasattr(e, 'response') else 'No details'}")
        except Exception as e:
            logger.error(f"Unexpected error (Attempt {attempt+1}): {type(e).__name__}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAY_SECONDS * (2 ** attempt))
            logger.info(f"Retrying LLM call for {target_model.__name__}...")
        else:
            logger.error(f"All {MAX_RETRIES} attempts failed for {target_model.__name__}")
            # Don't raise HTTPException here - let the caller handle it
            raise Exception(f"Failed to generate valid {target_model.__name__} after {MAX_RETRIES} attempts")


async def generate_draft(topic: str) -> AICourseDraft:
    """Generates the full course draft using a single LLM call with structured output."""
    
    # Create a comprehensive prompt that includes the JSON schema
    prompt = f"""
Create a complete course draft for the topic: "{topic}"

You must return a JSON object with exactly this structure (no additional text):

{{
  "outline": {{
    "title": "Course title here",
    "sections": ["Section 1", "Section 2", "Section 3", "Section 4", "Section 5"]
  }},
  "lesson": {{
    "objectives": ["Learning objective 1", "Learning objective 2", "Learning objective 3"],
    "examples": ["Real-world example 1", "Real-world example 2", "Real-world example 3"],
    "mini_project": "Detailed mini-project description suitable for Grade 11 students"
  }},
  "mcqs": [
    {{
      "stem": "Multiple choice question 1?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option A"
    }},
    {{
      "stem": "Multiple choice question 2?", 
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option B"
    }}
  ],
  "saqs": [
    {{
      "stem": "Short answer question 1?",
      "correct_answer": "Expected short answer"
    }},
    {{
      "stem": "Short answer question 2?",
      "correct_answer": "Expected short answer 2"
    }}
  ]
}}

Requirements:
- Include exactly 10 MCQs and 3 SAQs (I've shown 2 MCQs and 2 SAQs as examples above)
- All content should be appropriate for Grade 11 reading level
- MCQs must have exactly 4 options each
- Questions should test understanding, not just memorization  
- Mini-project should be practical and engaging
- All strings must be properly escaped for JSON
- Return ONLY the JSON object, no explanations or markdown
"""
    
    try:
        logger.info(f"Starting draft generation for topic: {topic}")
        draft = await call_openai_structured(prompt, AICourseDraft, settings.ai_model_name)
        logger.info(f"Successfully generated course draft for topic: {topic}")
        
        # Log some basic info about what was generated
        logger.info(f"Generated draft with {len(draft.mcqs)} MCQs and {len(draft.saqs)} SAQs")
        return draft
        
    except Exception as e:
        error_msg = f"Failed to generate course draft for '{topic}': {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500, 
            detail=error_msg
        )


async def review_draft(draft: AICourseDraft) -> AIReviewFeedback:
    """Performs an AI and automated review pass on the generated draft."""
    content_for_review = draft.model_dump_json(indent=2)
    
    review_prompt = f"""
Review this course draft and provide feedback in JSON format:

{content_for_review}

Return a JSON object with this structure:
{{
  "overall_quality_score": 85,
  "strengths": ["Strength 1", "Strength 2"],
  "improvement_areas": ["Area 1", "Area 2"],
  "specific_feedback": {{
    "outline_feedback": "Feedback about the outline",
    "lesson_feedback": "Feedback about the lesson",
    "mcq_feedback": "Feedback about MCQs", 
    "saq_feedback": "Feedback about SAQs"
  }},
  "reading_level_mismatch": null,
  "duplicate_flags": []
}}

Provide constructive feedback focusing on educational effectiveness, clarity, and Grade 11 appropriateness.
Return ONLY the JSON object.
"""

    ai_review_feedback = await call_openai_structured(
        review_prompt, 
        AIReviewFeedback, 
        settings.ai_review_model_name
    )
    
    # Add automated checks
    reading_level_mismatch = await check_reading_level(draft.lesson.mini_project)
    duplicate_flags = await check_duplicates(draft)

    if reading_level_mismatch:
        ai_review_feedback.reading_level_mismatch = reading_level_mismatch
    if duplicate_flags:
        ai_review_feedback.duplicate_flags = duplicate_flags

    return ai_review_feedback


async def check_reading_level(text: str) -> str | None:
    """Check reading level against external API."""
    if not settings.reading_level_api_url:
        return None
        
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                settings.reading_level_api_url, 
                json={"text": text}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("mismatch_reason")
                else:
                    logger.warning(f"Reading level API returned status {response.status}")
        except aiohttp.ClientError as e:
            logger.error(f"Error calling reading level API: {e}")
    return None


async def check_duplicates(draft: AICourseDraft) -> list[str]:
    """Check for duplicates against external API."""
    if not settings.duplicate_detector_api_url:
        return []
        
    all_text = [mcq.stem for mcq in draft.mcqs] + [saq.stem for saq in draft.saqs] + draft.lesson.examples
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                settings.duplicate_detector_api_url, 
                json={"texts": all_text}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("duplicate_flags", [])
                else:
                    logger.warning(f"Duplicate detector API returned status {response.status}")
        except aiohttp.ClientError as e:
            logger.error(f"Error calling duplicate detector API: {e}")
    return []


async def publish_to_sensai(draft: AICourseDraft) -> dict:
    """Publishes the final draft to the SensAI API."""
    headers = {
        "Authorization": f"Bearer {settings.sensai_publish_api_key}", 
        "Content-Type": "application/json"
    }
    payload = draft.model_dump_json()

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                settings.sensai_publish_api_url, 
                data=payload, 
                headers=headers
            ) as response:
                response_body = await response.json()
                if response.status >= 400:
                    logger.error(f"SensAI Publish API error {response.status}: {response_body}")
                return {"status_code": response.status, "body": response_body}
        except aiohttp.ClientError as e:
            logger.error(f"Error calling SensAI Publish API: {e}")
            return {
                "status_code": 500, 
                "body": {"detail": f"Failed to connect to publish API: {e}"}
            }