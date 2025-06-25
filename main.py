from fastapi import FastAPI, Request
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
            "You are a strict NCERT question source checker. Do not guess."
            " Only say 'Yes' if the content is present in the source."
            " Respond with: 'Yes' or 'No'."
            " If 'Yes', also include Chapter, Topic, Subtopic."
            " If 'No', say only 'No'."
        ),
        input=input_messages,
        tools=tools,
        include=["file_search_call.results"]
    )

    return {"answer": response.output_text.strip()}
