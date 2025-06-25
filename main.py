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
            "You are a strict NCERT checker. Only answer 'Yes' if the content is present in the provided files."
            " Say 'No' if there is no matching content. Never guess or use general knowledge."
        ),
        input=input_messages,
        tools=tools,
        include=["file_search_call.results"]
    )

    file_search_results = []

    for output_item in response.output:
        if output_item.type == "file_search_call":
            file_search_results = output_item.results
            for result in file_search_results:
                print("Score:", result.score)
                print("Snippet:", result.text[:200].replace('\n', ' '))
                if result.score > 0.7:
                    return {"answer": "Yes"}

    return {"answer": "No"}

    
