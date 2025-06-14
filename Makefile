.PHONY: dev deploy clean test

test:
	python test/ws_client.py \
		--url ws://localhost:8000/ws/stream \
		--file path/to/audio.wav \
		--chunk-duration 2.0

dev:
	docker compose up --build --force-recreate --remove-orphans --detach

deploy:
	git commit --allow-empty -m "CI Trigger: $(shell date +%Y-%m-%dT%H:%M:%S)"
	git push ci

clean:
	docker compose down --volumes --remove-orphans
