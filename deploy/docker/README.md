# Docker deployment

This single-host Compose deployment runs PostgreSQL, the Pelagia API and CPU
worker, PelagiaView, and a local HTTP proxy. Oracle Builder keeps its existing
Docker Compose stack, including its orchestrator, private runtime workers, and
separate durable data root. Both stacks use the `oracle-builder` Docker network.
The browser reaches Pelagia through `/api`; only Pelagia calls Oracle Builder.

## Prerequisites

- Docker Engine with Compose v2.17+ and BuildKit (`docker compose version`).
- Sibling checkouts named `Pelagia`, `PelagiaView`, and `oracle-builder` in the
  same parent directory. The build uses Oracle's local `oracle_data_contracts`.
- Sufficient disk space for PostgreSQL, image blobs, and Oracle model artifacts.

Start and configure Oracle Builder using
[`oracle-builder/deploy/docker/README.md`](../../../oracle-builder/deploy/docker/README.md).
Start its Compose stack before the commands below. It creates the
`oracle-builder` network; Pelagia joins that existing network by default, and
the Oracle worker deployment profile uses the same network name. Check it with
`docker network inspect oracle-builder`. Create an Oracle
analyst-or-higher token for Pelagia's V2 inference requests. Oracle's Web GUI
operator token is a different credential and should not be reused.

From this directory:

```bash
cp .env.example .env
chmod 600 .env
# Edit .env: set a random URL-safe POSTGRES_PASSWORD and the Oracle API token.
docker compose up -d postgres
docker compose build api worker view
docker compose run --rm api pelagia init-system
docker compose up -d api worker view proxy
docker compose ps
curl --fail http://127.0.0.1:8080/api/health
```

Open `http://127.0.0.1:8080/`. Create the first admin account once:

```bash
read -s 'ADMIN_PASSWORD?Admin password: '; echo
docker compose exec -e ADMIN_PASSWORD="$ADMIN_PASSWORD" api sh -c \
  'pelagia create-user YOUR_USERNAME --password "$ADMIN_PASSWORD" --admin'
unset ADMIN_PASSWORD
```

Pelagia's first login is projectless; create a project in PelagiaView or with
the API, then switch into it. An
existing browser session may retain an older API URL; set it to `/api` in the
connection screen if needed.

The HTTP proxy binds to loopback by default. For remote access, set
`PELAGIA_BIND_ADDRESS` to a suitable interface and put the stack behind a TLS
reverse proxy. Keep PostgreSQL and the Oracle orchestrator
ports private. The Oracle stack's own Caddy listener occupies ports 80/443.

## Upgrades and data

### Development channel (default)

All three repositories use `main` as the moving development channel. After
the initial installation, run this from the Pelagia checkout to update all
three to the latest `origin/main` and redeploy them:

```bash
python3 deploy/docker/update-dev.py
```

The updater requires clean `main` checkouts and both Docker stacks already
running. It fetches all three branches and verifies they can fast-forward
before changing any checkout. It builds the new images while services are
running, refuses to interrupt an active Oracle lease, waits for the Pelagia
worker to stop, stops the idle Oracle Docker deployments, migrates the Pelagia
database, restarts services, checks health, and starts the same Oracle
deployments from the new worker image. Do not start new jobs during an update.
The updater records the exact commit SHAs and image IDs in
`deploy/docker/runtime/dev-deployment.json` on the server. Run
`python3 deploy/docker/update-dev.py --check` for a local checkout check.

Back up PostgreSQL and `pelagia_data` together, plus Oracle's separate host
data root, before an update that may change persistent formats. The updater
does not make a backup or roll back a database migration. If a step fails after
services begin stopping, it leaves `deploy/docker/runtime/dev-update.json`
with the Oracle deployment IDs and stops. Inspect the error and service logs
before restarting; remove that state file only after recovery. The updater
does not change a dirty checkout or merge a divergent branch.

Coordinated releases can later use pinned image tags and the same migration
and health checks. Do not use a floating image tag as the record of what ran;
the development manifest supplies that record now.

Do not use `docker compose down -v`: it deletes the named PostgreSQL and
Pelagia data volumes. Mount an import directory into both `api` and `worker`
at `/var/lib/pelagia/import` if users need to ingest files already on the
host. This deployment runs one CPU worker by default; scale it with
`docker compose up -d --scale worker=N` after checking CPU, memory, and
PostgreSQL connection capacity. Oracle Builder manages its own inference
workers and accelerator placement.

The images are built from the current sibling checkouts. Verify a small ingest
and inference job after upgrading.
