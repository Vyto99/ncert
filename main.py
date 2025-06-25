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
        "max_num_results": 3
    }]

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=(
            "You're a strict NCERT checker. ONLY answer 'Yes' if the exact concept or line exists in the files."
            " If not present, answer 'No'. If 'Yes', return Chapter, Topic, Subtopic from the matched content."
            " Never guess or use external knowledge. Use only the provided vector search results."
        ),
        input=input_messages,
        tools=tools,
        include=["file_search_call.results"]
    )

    # Extract and evaluate file_search results
    found = False
    matched_snippet = ""
    file_search_results = []

    for output_item in response.output:
        if output_item.type == "file_search_call":
            file_search_results = output_item.results
            if file_search_results:
                found = True
                matched_snippet = file_search_results[0].text.strip()

    if not found:
        return {"answer": "No"}

    # Optionally: you can try to extract metadata like Chapter/Topic/Subtopic
    # For now, we assume it's embedded in the text
    return {
        "answer": "Yes",
        "matched_text": matched_snippet[:500]  # Optional preview
    }
