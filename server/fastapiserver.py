from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn
import pandas as pd
from model.search_engine import SearchEngine

app = FastAPI()


@app.post("/search-news")
async def search_news_in_file(query: str = Form(...), file: UploadFile = File(...)):
    if file is None:
        raise HTTPException(status_code=400, detail="Файл не был загружен.")

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения файла CSV: {e}")

    if 'text' not in df.columns or 'labels' not in df.columns:
        raise HTTPException(status_code=400, detail="CSV файл должен содержать столбцы 'text' и 'labels'.")

    if not query:
        raise HTTPException(status_code=400, detail="Параметр 'query' не должен быть пустым.")

    model = SearchEngine(text_column='text', id_column='labels')
    model.fit(df, perform_stem=False)

    results = model.get_results(query)
    return {"results": results.to_dict(orient='records')}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
