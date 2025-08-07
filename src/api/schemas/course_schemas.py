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

# Schema for the AI review feedback
class AIReviewFeedback(BaseModel):
    ambiguity_flags: list[str] = Field(default_factory=list)
    bias_flags: list[str] = Field(default_factory=list)
    reading_level_mismatch: str | None = None
    duplicate_flags: list[str] = Field(default_factory=list)

# Schema for the full response from the AI assistant endpoint
class AIResponse(BaseModel):
    draft: AICourseDraft
    review: AIReviewFeedback
    publish_status: dict | None = None

# This is a new schema for the SensAI publish endpoint
class PublishResponse(BaseModel):
    message: str
    published_at: datetime
    course_title: str