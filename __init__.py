from flask import Flask

def create_app():
    app = Flask(__name__)
    app.secret_key = 'your_secret_key'  # Optional, for session management

    from .routes import main
    app.register_blueprint(main)  # Register the blueprint

    return app
