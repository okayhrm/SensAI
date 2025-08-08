import openai
from pydantic import ValidationError, BaseModel
import aiohttp
import json
import asyncio
from typing import List, Dict, Any, TypeVar, Type, Union, Optional
import os
from fastapi import HTTPException
import re

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

# New schemas for validation
class TopicValidation(BaseModel):
    is_educational: bool
    is_specific: bool
    ambiguity_flags: List[str]
    bias_flags: List[str]
    suggested_clarifications: List[str]
    topic_category: str  # e.g., "programming", "science", "history", etc.

class ContentBiasCheck(BaseModel):
    has_bias: bool
    bias_types: List[str]  # e.g., ["gender", "cultural", "age", "socioeconomic"]
    problematic_content: List[str]
    severity_score: int  # 1-10 scale

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

async def validate_topic(topic: str) -> TopicValidation:
    """Validates the input topic for educational appropriateness, specificity, and potential issues."""
    
    validation_prompt = f"""
Analyze this topic for course creation: "{topic}"

Evaluate and return a JSON response with this structure:
{{
  "is_educational": true/false,
  "is_specific": true/false,
  "ambiguity_flags": ["flag1", "flag2"],
  "bias_flags": ["bias_type1", "bias_type2"],
  "suggested_clarifications": ["suggestion1", "suggestion2"],
  "topic_category": "category_name"
}}

Evaluation criteria:
1. is_educational: Can this be turned into an educational course appropriate for Grade 11?
2. is_specific: Is the topic specific enough, or too broad/vague?
3. ambiguity_flags: List issues like ["too_vague", "multiple_interpretations", "unclear_scope"]
4. bias_flags: Check for potential biases like ["cultural", "gender", "religious", "political", "socioeconomic"]
5. suggested_clarifications: What clarifying questions should be asked?
6. topic_category: Classify into categories like "programming", "science", "history", "language", "arts", etc.

Examples of problems:
- "Python" is ambiguous (programming language vs snake?)
- "Java" could be programming or geography
- Topics with political/religious bias need flagging
- Overly broad topics like "science" need specificity

Return ONLY the JSON object.
"""
    
    try:
        validation = await call_openai_structured(validation_prompt, TopicValidation, settings.ai_model_name)
        logger.info(f"Topic validation completed for: {topic}")
        return validation
    except Exception as e:
        logger.error(f"Failed to validate topic '{topic}': {str(e)}")
        # Return a safe default that will block course creation
        return TopicValidation(
            is_educational=False,
            is_specific=False,
            ambiguity_flags=["validation_failed"],
            bias_flags=[],
            suggested_clarifications=["Please provide a more specific topic"],
            topic_category="unknown"
        )

async def check_content_bias(draft: AICourseDraft) -> ContentBiasCheck:
    """Checks the generated content for bias issues."""
    
    # Combine all content for bias checking
    all_content = {
        "title": draft.outline.title,
        "sections": draft.outline.sections,
        "objectives": draft.lesson.objectives,
        "examples": draft.lesson.examples,
        "mini_project": draft.lesson.mini_project,
        "mcq_questions": [mcq.stem for mcq in draft.mcqs],
        "mcq_options": [option for mcq in draft.mcqs for option in mcq.options],
        "saq_questions": [saq.stem for saq in draft.saqs]
    }
    
    bias_check_prompt = f"""
Analyze this course content for bias and problematic representations:

{json.dumps(all_content, indent=2)}

Check for various types of bias and return JSON:
{{
  "has_bias": true/false,
  "bias_types": ["type1", "type2"],
  "problematic_content": ["specific example 1", "specific example 2"],
  "severity_score": 1-10
}}

Look for these bias types:
- gender: Male-dominated examples, stereotypical roles
- cultural: Western-centric examples, cultural assumptions
- socioeconomic: Assumptions about resources, lifestyle
- age: Ageist language or assumptions
- racial: Racial stereotypes or exclusion
- ability: Ableist language or assumptions
- religious: Religious bias or assumptions
- geographic: Urban/rural bias, country-specific assumptions

Severity scale:
1-3: Minor bias, easily correctable
4-6: Moderate bias, needs attention
7-10: Severe bias, major revision needed

Return ONLY the JSON object.
"""
    
    try:
        bias_check = await call_openai_structured(bias_check_prompt, ContentBiasCheck, settings.ai_model_name)
        logger.info(f"Bias check completed - Has bias: {bias_check.has_bias}")
        return bias_check
    except Exception as e:
        logger.error(f"Failed to check content bias: {str(e)}")
        # Return a safe default that flags for manual review
        return ContentBiasCheck(
            has_bias=True,
            bias_types=["check_failed"],
            problematic_content=["Bias check failed - manual review required"],
            severity_score=5
        )

async def generate_draft(topic: str) -> tuple[AICourseDraft, TopicValidation, ContentBiasCheck]:
    """Generates the full course draft with validation and bias checking."""
    
    # Step 1: Validate the topic
    logger.info(f"Validating topic: {topic}")
    topic_validation = await validate_topic(topic)
    
    # Check if topic is suitable for course creation
    # Only block truly inappropriate topics, flag everything else as warnings
    if not topic_validation.is_educational and "inappropriate" in topic_validation.bias_flags:
        raise HTTPException(
            status_code=400,
            detail=f"Topic '{topic}' is not suitable for educational course creation."
        )
    
    # Convert validation issues to warnings instead of hard errors
    validation_warnings = []
    
    if not topic_validation.is_educational:
        validation_warnings.append("Topic may not be fully educational - proceeding with caution")
        logger.warning(f"Topic '{topic}' flagged as potentially non-educational")
    
    if not topic_validation.is_specific:
        validation_warnings.append(f"Topic is vague - attempting to infer specifics. Suggestions: {topic_validation.suggested_clarifications}")
        logger.warning(f"Topic '{topic}' is vague, but attempting generation")
    
    if topic_validation.bias_flags:
        validation_warnings.append(f"Potential topic bias detected: {topic_validation.bias_flags}")
        logger.warning(f"Topic has potential bias flags: {topic_validation.bias_flags}")
    
    if topic_validation.ambiguity_flags:
        validation_warnings.append(f"Topic ambiguity detected: {topic_validation.ambiguity_flags}")
        logger.warning(f"Topic has ambiguity flags: {topic_validation.ambiguity_flags}")
    
    # Store warnings for later use
    topic_warnings = validation_warnings
    
    # Step 2: Generate course with enhanced prompt that handles vague topics
    enhanced_prompt = f"""
Create a complete course draft for the topic: "{topic}"

IMPORTANT CONTEXT:
- Topic category: {topic_validation.topic_category}
- This topic was validated as educational: {topic_validation.is_educational}
- Topic specificity: {topic_validation.is_specific}
- Validation warnings: {topic_warnings if topic_warnings else "None"}

HANDLING VAGUE TOPICS:
- If the topic is vague (like "python"), make reasonable assumptions about the most educational interpretation
- For "python", assume "Python programming language fundamentals"
- For "java", assume "Java programming language basics" 
- For "math", assume "Mathematics fundamentals for Grade 11"
- Focus on the most common educational interpretation
- Include a note in the course title clarifying the interpretation

BIAS PREVENTION GUIDELINES:
- Use diverse, inclusive examples from various cultures and backgrounds
- Avoid gender stereotypes in examples and scenarios
- Include examples accessible to different socioeconomic backgrounds
- Use universal concepts that don't assume specific cultural knowledge
- Ensure examples represent different regions/countries when relevant
- Use neutral, inclusive language throughout

You must return a JSON object with exactly this structure (no additional text):

{{
  "outline": {{
    "title": "Course title here (clarify if topic was vague, e.g., 'Python Programming Language Basics')",
    "sections": ["Section 1", "Section 2", "Section 3", "Section 4", "Section 5"]
  }},
  "lesson": {{
    "objectives": ["Learning objective 1", "Learning objective 2", "Learning objective 3"],
    "examples": ["Real-world example 1", "Real-world example 2", "Real-world example 3"],
    "mini_project": "Detailed mini-project description suitable for Grade 11 students"
  }},
  "mcqs": [
    {{
      "stem": "Multiple choice question 1 about {topic}?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option A"
    }},
    {{
      "stem": "Multiple choice question 2 about {topic}?", 
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option B"
    }}
  ],
  "saqs": [
    {{
      "stem": "Short answer question 1 about {topic}?",
      "correct_answer": "Expected short answer"
    }},
    {{
      "stem": "Short answer question 2 about {topic}?",
      "correct_answer": "Expected short answer 2"
    }}
  ]
}}

Requirements:
- The course MUST be specifically about the most educational interpretation of "{topic}"
- If topic is vague, make the title specific (e.g., "Python" → "Python Programming Language Fundamentals")
- Include exactly 10 MCQs and 3 SAQs (I've shown 2 MCQs and 2 SAQs as examples above)
- All content should be appropriate for Grade 11 reading level
- MCQs must have exactly 4 options each
- Questions should test understanding, not just memorization  
- Mini-project should be practical and engaging
- Use diverse, inclusive examples that avoid cultural, gender, or socioeconomic bias
- All strings must be properly escaped for JSON
- Return ONLY the JSON object, no explanations or markdown
"""
    
    try:
        logger.info(f"Starting draft generation for validated topic: {topic}")
        draft = await call_openai_structured(enhanced_prompt, AICourseDraft, settings.ai_model_name)
        logger.info(f"Successfully generated course draft for topic: {topic}")
        
        # Step 3: Check the generated content for bias
        logger.info("Checking generated content for bias...")
        bias_check = await check_content_bias(draft)
        
        # If severe bias detected, reject the draft
        if bias_check.has_bias and bias_check.severity_score >= 7:
            raise HTTPException(
                status_code=500,
                detail=f"Generated content has severe bias issues: {bias_check.bias_types}. Please try again with a different approach."
            )
        
        # Log warnings for moderate bias
        if bias_check.has_bias and bias_check.severity_score >= 4:
            logger.warning(f"Generated content has moderate bias: {bias_check.bias_types}")
            logger.warning(f"Problematic content: {bias_check.problematic_content}")
        
        # Log some basic info about what was generated
        logger.info(f"Generated draft with {len(draft.mcqs)} MCQs and {len(draft.saqs)} SAQs")
        return draft, topic_validation, bias_check, topic_warnings
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        error_msg = f"Failed to generate course draft for '{topic}': {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500, 
            detail=error_msg
        )

async def review_draft(draft: AICourseDraft, topic_validation: TopicValidation, bias_check: ContentBiasCheck) -> AIReviewFeedback:
    """Performs an enhanced AI and automated review pass on the generated draft."""
    content_for_review = draft.model_dump_json(indent=2)
    
    review_prompt = f"""
Review this course draft with additional context about validation and bias checking:

COURSE CONTENT:
{content_for_review}

VALIDATION CONTEXT:
- Topic category: {topic_validation.topic_category}
- Original ambiguity flags: {topic_validation.ambiguity_flags}
- Original bias flags: {topic_validation.bias_flags}

BIAS CHECK RESULTS:
- Has bias: {bias_check.has_bias}
- Bias types found: {bias_check.bias_types}
- Severity score: {bias_check.severity_score}/10
- Problematic content: {bias_check.problematic_content}

Return a JSON object with this structure:
{{
  "overall_quality_score": 85,
  "strengths": ["Strength 1", "Strength 2"],
  "improvement_areas": ["Area 1", "Area 2"],
  "bias_addressed": true,
  "inclusivity_score": 8,
  "ambiguity_flags": [],
  "bias_flags": [],
  "reading_level_mismatch": null,
  "duplicate_flags": []
}}

Focus your review on:
1. Educational effectiveness and Grade 11 appropriateness
2. Whether bias concerns were properly addressed
3. Inclusivity and diversity of examples
4. Content accuracy and relevance to the specific topic
5. Clear learning outcomes and practical application

Return ONLY the JSON object.
"""

    try:
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
        
    except Exception as e:
        logger.error(f"Failed to review draft: {str(e)}")
        # Return a basic review structure
        from api.schemas.course_schemas import AIReviewFeedback
        return AIReviewFeedback(
            overall_quality_score=50,
            strengths=["Course generated successfully"],
            improvement_areas=["Review failed - manual review recommended"],
            specific_feedback={
                "outline_feedback": "Review failed",
                "lesson_feedback": "Review failed", 
                "mcq_feedback": "Review failed",
                "saq_feedback": "Review failed"
            },
            reading_level_mismatch=None,
            duplicate_flags=[]
        )

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