.PHONY: install run test docker-build docker-run tunnel

install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q

docker-build:
	docker build -t gridwise:latest .

docker-run:
	docker run --rm -p 8000:8000 -e GEMINI_API_KEY=$$GEMINI_API_KEY gridwise:latest

tunnel:
	cloudflared tunnel --no-autoupdate --url http://localhost:8000
