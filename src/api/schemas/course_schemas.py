from pydantic import BaseModel, Field, conlist
from datetime import datetime

# Schema for the Course Outline
class CourseOutline(BaseModel):
    title: str = Field(..., description="The title of the course outline.")
    sections: list[str] = Field(..., description="A list of section titles for the course.")

# Schema for a single Multiple Choice Question
class MCQ(BaseModel):
    stem: str = Field(..., description="The body of the question.")
    options: list[str] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="A list of possible answers."
    )
    correct_answer: str = Field(..., description="The correct answer from the options.")

# Schema for a single Short Answer Question
class SAQ(BaseModel):
    stem: str = Field(..., description="The body of the short answer question.")
    correct_answer: str = Field(..., description="The expected correct answer or key points.")

# Schema for the Lesson Draft
class LessonDraft(BaseModel):
    objectives: list[str] = Field(..., description="Learning objectives for the lesson.")
    examples: list[str] = Field(..., description="Examples to illustrate key concepts.")
    mini_project: str = Field(..., description="A description of a mini-project for students to complete.")

# Schema for the full AI-generated course content
class AICourseDraft(BaseModel):
    outline: CourseOutline
    lesson: LessonDraft
    mcqs: list[MCQ]
    saqs: list[SAQ]

# NEW: Add these schemas for enhanced validation
class TopicValidation(BaseModel):
    is_educational: bool
    is_specific: bool
    ambiguity_flags: list[str] = Field(default_factory=list)
    bias_flags: list[str] = Field(default_factory=list)
    suggested_clarifications: list[str] = Field(default_factory=list)
    topic_category: str

class ContentBiasCheck(BaseModel):
    has_bias: bool
    bias_types: list[str] = Field(default_factory=list)
    problematic_content: list[str] = Field(default_factory=list)
    severity_score: int = Field(..., ge=1, le=10, description="Bias severity score from 1-10")

# UPDATED: Enhanced AI review feedback (replace your existing one)
class AIReviewFeedback(BaseModel):
    ambiguity_flags: list[str] = Field(default_factory=list)
    bias_flags: list[str] = Field(default_factory=list)
    reading_level_mismatch: str | None = None
    duplicate_flags: list[str] = Field(default_factory=list)
    # New enhanced fields
    overall_quality_score: int = Field(default=75, ge=0, le=100)
    strengths: list[str] = Field(default_factory=list)
    improvement_areas: list[str] = Field(default_factory=list)
    bias_addressed: bool = Field(default=True)
    inclusivity_score: int = Field(default=7, ge=1, le=10)

# UPDATED: Enhanced AI response (replace your existing one)
class AIResponse(BaseModel):
    draft: AICourseDraft
    review: AIReviewFeedback
    publish_status: dict | None = None
    # New fields for enhanced validation
    topic_validation: TopicValidation | None = None
    bias_check: ContentBiasCheck | None = None
    warnings: list[str] = Field(default_factory=list)

# This is a new schema for the SensAI publish endpoint
class PublishResponse(BaseModel):
    message: str
    published_at: datetime
    course_title: str