# api/index.py
from vercel_python_wsgi import wsgi_handler
from app import app

handler = wsgi_handler(app)