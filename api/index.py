from app import app
from flask import request

def handler(request, context):
    with app.request_context(request.environ):
        try:
            response = app.full_dispatch_request()
            return {
                'statusCode': response.status_code,
                'headers': dict(response.headers),
                'body': response.get_data().decode('utf-8')
            }
        except Exception as e:
            return {
                'statusCode': 500,
                'body': f'Internal Server Error: {str(e)}'
            }