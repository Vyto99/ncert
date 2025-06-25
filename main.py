from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

class QuestionInput(BaseModel):
    question: str

@app.post("/check-question")
async def check_question(data: QuestionInput):
    input_messages = [{"role": "user", "content": data.question}]
    
    tools = [{
        "type": "file_search",
        "vector_store_ids": ["vs_685c46f97c8c8191801c6489358e70b9"],
        "max_num_results": 5
    }]

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=(
            "You're a strict NCERT checker. ONLY answer 'Yes' if the exact concept or line exists in the files."
            " If not present, answer 'No'. If 'Yes', extract Chapter, Topic, and Subtopic from the text or filename."
            " Never guess or use external knowledge. Use only the provided vector search results."
        ),
        input=input_messages,
        tools=tools,
        include=["file_search_call.results"]
    )

    matched = None
    file_search_results = []

    for output_item in response.output:
        if output_item.type == "file_search_call":
            file_search_results = output_item.results
            for result in file_search_results:
                print("Score:", result.score)
                print("Snippet:", result.text[:200].replace('\n', ' '))
                if result.score > 0.7:
                    matched = result
                    break  # Stop at first relevant match

    if not matched:
        return {"answer": "No"}

    # Try to extract context info
    snippet = matched.text.strip()

    # Simple placeholder parsing logic (customize based on your actual content format)
    chapter = topic = subtopic = "Not Found"
    if "chapter" in snippet.lower():
        chapter = snippet.splitlines()[0]
    if "topic" in snippet.lower():
        topic = snippet.splitlines()[1] if len(snippet.splitlines()) > 1 else "Not Found"
    if "subtopic" in snippet.lower():
        subtopic = snippet.splitlines()[2] if len(snippet.splitlines()) > 2 else "Not Found"

    return {
        "answer": "Yes",
        "chapter": chapter.strip(),
        "topic": topic.strip(),
        "subtopic": subtopic.strip(),
        "score": matched.score,
        "matched_text": snippet[:300] + "..." if len(snippet) > 300 else snippet
    }
