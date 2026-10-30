"""Gunicorn and local Flask entry point."""

from ethereum_fraud.web import create_app

app = create_app()
