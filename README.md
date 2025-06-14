# Classify API

A scalable and robust API service for audio classification, designed to detect specific species from audio inputs. This service is built with a modern Python stack, featuring a FastAPI web server, Celery for asynchronous task processing, and a Redis message broker. The entire application is containerized with Docker for easy and reproducible deployments.

## Configuration

Application configuration, including authorized clients and their permissions, is managed in src/config.yml.

### Client Configuration

Clients are defined under the authorized_clients key. Each client has a descriptive key and the following properties:

- name: The human-readable name of the client.
- token: The secret API key used for authentication.
- enabled: A boolean (true or false) to enable or disable the client.
- scopes: A list of permissions granted to the client.

Example `src/config.yml`:

```yaml
authorized_clients:
  classify-web:
    name: "Humbug Classification Web Interface"
    token: "54d37910-2a16-46fa-b68d-3f39fd4b98e8"
    enabled: true
    scopes:
      - "predict:med"
      - "predict:msc"
      - "models:read"

  legacy-client:
    name: "Legacy Client"
    token: "a1b2c3d4-e5f6-7890-1234-567890abcdef"
    enabled: false
    scopes:
      - "predict:med"
```

## Installation & Running

This project is designed to be run with Docker and Docker Compose.

Prerequisites

- Docker
- Docker Compose
- Running the Application
- Clone the repository.

Navigate to the project's root directory.

Start the services using Docker Compose:

```bash
docker-compose up --build
```

- The --build flag is only necessary the first time you run the application or after changing dependencies.
- This command will build the Python application image and start the api, worker, and redis containers.
