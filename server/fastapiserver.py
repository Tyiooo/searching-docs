from fastapi import FastAPI, UploadFile, HTTPException, Form, File, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
from model.search_engine import SearchEngine
from typing import Optional


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: Optional[SearchEngine] = None


@app.post("/initiate-news")
def initiate_news(id_column: str = Form(...), text_column: str = Form(...), file: UploadFile = File(...)):
    global model
    if file is None:
        raise HTTPException(status_code=400, detail="Файл не был загружен.")

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения файла CSV: {e}")

    model = SearchEngine(text_column=text_column, id_column=id_column)
    model.fit(df, perform_stem=False)
    return {"message": "Документ инициализирован успешно"}


@app.get("/search-news")
def search_news(query: str = Query(...)):
    if model is None:
        raise HTTPException(status_code=400, detail="Документ не был инициализирован.")
    if not query:
        raise HTTPException(status_code=400, detail="Параметр 'query' не должен быть пустым.")

    results = model.get_results(query)
    return {"results": results.to_dict(orient='records')}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
