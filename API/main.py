from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from inference import predict_credibility
from fastapi.middleware.cors import CORSMiddleware

# pydantic is used for request bodies
class News(BaseModel):
    """Declares a JSON object to be used from request"""
    text: str # article text

app = FastAPI()

# use CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to FASTAPI"}

@app.post("/credibility/")
async def credibility(news: News):
    credibility_score = predict_credibility(news=[news.text])
    return {'credibility_score': credibility_score}

if __name__ == 'main':
    uvicorn.run(app, host='0.0.0.0', port=80)
    # app available at localhost:80

# uvicorn main:app --host 0.0.0.0 --port 80 (How to run on port 80)
# fastapi run --workers 4 main.