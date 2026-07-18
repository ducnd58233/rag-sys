from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from src.shared.configs.settings import Settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def database_url() -> str:
    return Settings().database.url.get_secret_value()


def run_migrations_offline() -> None:
    context.configure(
        url=database_url(),
        literal_binds=True,
        dialect_opts={'paramstyle': 'named'},
        compare_type=False,
        target_metadata=None,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        compare_type=False,
        target_metadata=None,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section['sqlalchemy.url'] = database_url()

    engine = async_engine_from_config(
        section,
        prefix='sqlalchemy.',
        poolclass=pool.NullPool,
    )

    async with engine.connect() as connection:
        await connection.run_sync(run_migrations)

    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()