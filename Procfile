web: uvicorn app:app --host 0.0.0.0 --port $PORT --workers $WEB_CONCURRENCY --limit-max-requests 500 --limit-concurrency 50 --timeout-keep-alive 30
release: python -c "import nltk; nltk.download()"