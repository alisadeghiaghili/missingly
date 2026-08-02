# Database migrations

`main.py` owns a small, idempotent schema migration layer so a fresh deployment can boot safely on either local SQLite or production PostgreSQL.

- `0001_production_foundation` creates the durable users, billing, usage, session, project, audit, and settings tables plus the required indexes.
- `0002_collaboration_publishing` adds project roles and immutable published-site snapshots.
- Every applied version is stored in `schema_migrations`.
- PostgreSQL is the only allowed production database. Set `DATABASE_URL` before the first production boot; the application validates this at startup.

Before applying a future schema version, back up the production database and test the version against a production-sized restore. Add a new, idempotent migration to `create_postgres_tables` and its SQLite compatibility path, then add regression coverage before release.
