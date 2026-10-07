FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# baixa o modelo de reranking na build (evita download no primeiro uso)
RUN python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MultiBERT-L-12', cache_dir='/models')"
COPY app app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
