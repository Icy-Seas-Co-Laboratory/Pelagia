# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# The contracts package is maintained in the sibling Oracle Builder repository.
COPY --from=oracle_contracts . /tmp/oracle_data_contracts
RUN pip install --no-cache-dir /tmp/oracle_data_contracts
COPY . /app
RUN pip install --no-cache-dir '.[api,cli,postgres,kvstore-blake3,worker-cpu]'
RUN mkdir -p /var/lib/pelagia/import

CMD ["uvicorn", "Pelagia.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
