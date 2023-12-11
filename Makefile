.PHONY: run-prod down-prod run-dev down-dev

run-prod:
	docker-compose -f docker-compose.prod.yaml up

down-prod:
	docker-compose -f docker-compose.prod.yaml down

run-dev:
	docker-compose -f docker-compose.dev.yaml up

down-dev:
	docker-compose -f docker-compose.dev.yaml down