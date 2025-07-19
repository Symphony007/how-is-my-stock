web: gunicorn --worker-tmp-dir /dev/shm --workers 2 --threads 4 --bind 0.0.0.0:$PORT app:app
release: python -c "from app import app, db; with app.app_context(): db.create_all()"