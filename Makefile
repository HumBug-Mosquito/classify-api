
.PHONY: dev prod clean test

test: 
	python test/ws_client.py \
		--url ws://localhost:8000/ws/stream \
		--file path/to/audio.wav \
		--chunk-duration 2.0

dev:
    docker-compose up api

prod:
    API_TOKEN=${API_TOKEN} AUTHORIZED_CLIENTS=${AUTHORIZED_CLIENTS} \
      docker-compose up --build

clean:
    docker compose down --volumes --remove-orphans