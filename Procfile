web: gunicorn --worker-tmp-dir /dev/shm --workers 2 --threads 4 --bind 0.0.0.0:$PORT app:app
release: python -c "from app import db; db.create_all()" || echo "Skipping DB creation"