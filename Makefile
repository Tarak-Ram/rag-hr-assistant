.PHONY: install run test eval lint up

install:
	pip install -r requirements-dev.txt

run:
	uvicorn app.main:app --reload

test:
	pytest -q

eval:
	python -m evals.run_evals --gate

lint:
	ruff check .

up:
	cp -n .env.example .env || true
	docker compose up --build
