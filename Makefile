.PHONY: up down logs ps clean

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

ps:
	docker-compose ps

clean:
	docker-compose down -v

mock-api:
	python3 app/tools/mock_api.py

ingest:
	export PYTHONPATH=$$PYTHONPATH:. && python3 scripts/ingest_knowledge.py
