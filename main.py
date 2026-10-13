
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from config import settings
from explanation_module import explain_topic
from learning_path import get_learning_recommendations
from qna import answer_question
from quiz_module import generate_quiz
from summary_module import summarize_text


BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="EduGenie API",
    description="AI-powered educational learning assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=30000)
    level: str = Field(default="beginner", max_length=40)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    level: str = Field(default="beginner", max_length=40)


class QuizRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    count: int = Field(default=3, ge=1, le=10)
    level: str = Field(default="beginner", max_length=40)


class LearningPathRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=300)
    level: str = Field(default="beginner", max_length=40)
    weeks: int = Field(default=4, ge=1, le=52)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"app_name": settings.app_name},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "app": settings.app_name,
        "gemini_configured": bool(settings.gemini_api_key.strip()),
    }


async def execute_ai(operation, *args):
    try:
        return {"result": await operation(*args)}
    except Exception as exc:
        # Log full details in the server terminal, but do not expose
        # internal exception details to API clients.
        print(f"AI operation failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502,
            detail=(
                "The AI request failed. Check the server terminal, "
                "your Gemini API key, model configuration, and network."
            ),
        ) from exc


@app.post("/qa")
async def qa(payload: QuestionRequest):
    return await execute_ai(
        answer_question,
        payload.question,
        payload.level,
    )


@app.post("/explain")
async def explain(payload: TextRequest):
    return await execute_ai(
        explain_topic,
        payload.text,
        payload.level,
    )


@app.post("/quiz")
async def quiz(payload: QuizRequest):
    return await execute_ai(
        generate_quiz,
        payload.text,
        payload.count,
        payload.level,
    )


@app.post("/summarize")
async def summarize(payload: TextRequest):
    return await execute_ai(
        summarize_text,
        payload.text,
        payload.level,
    )


@app.post("/learn/recommendations")
async def learning_recommendations(
    payload: LearningPathRequest,
):
    return await execute_ai(
        get_learning_recommendations,
        payload.topic,
        payload.level,
        payload.weeks,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )
