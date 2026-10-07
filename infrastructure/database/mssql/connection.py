"""SQLAlchemy engine and session setup for the dedicated activity database."""

from dataclasses import dataclass, field
import os
import re

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session, sessionmaker

from .models import ActivityBase, GitHubActivityModel


def _environment_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be set to yes or no.")


ACTIVITY_DATABASE_NAME = "MultiAgentPlatform"


def _require_activity_database(database_name: str | None) -> None:
    if (
        not isinstance(database_name, str)
        or not database_name.strip()
        or database_name.strip().casefold() != ACTIVITY_DATABASE_NAME.casefold()
    ):
        raise ValueError("GitHub activity persistence must target the MultiAgentPlatform database.")


def _require_setting(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty.")


def _database_from_url(url: URL) -> str | None:
    database_names = [url.database] if url.database else []
    odbc_connection = url.query.get("odbc_connect")
    if odbc_connection is not None and not isinstance(odbc_connection, str):
        raise ValueError("The ODBC connection string must be valid text.")
    if isinstance(odbc_connection, str):
        matches = re.findall(
            r"(?:^|;)\s*(?:DATABASE|INITIAL\s+CATALOG)\s*=\s*(?:\{([^}]*)\}|([^;]*))",
            odbc_connection,
            re.IGNORECASE,
        )
        database_names.extend((braced or plain).strip() for braced, plain in matches)
        if not matches:
            raise ValueError("The ODBC connection string must specify its target database.")
    normalized = {name.strip().casefold() for name in database_names if name and name.strip()}
    if len(normalized) > 1:
        raise ValueError("Conflicting database names were supplied for GitHub activity persistence.")
    if not normalized:
        return None
    return next(name.strip() for name in database_names if name and name.strip())


@dataclass(frozen=True)
class DatabaseSettings:
    """Dedicated database settings; URL is excluded from repr to protect secrets."""

    server: str = "localhost"
    database: str = "MultiAgentPlatform"
    driver: str = "ODBC Driver 18 for SQL Server"
    trust_server_certificate: bool = True
    database_url: str | None = field(default=None, repr=False)

    @classmethod
    def from_environment(cls) -> "DatabaseSettings":
        """Load activity database configuration without logging its URL."""
        database_url = os.environ.get("GITHUB_ACTIVITY_DATABASE_URL") or None
        server = os.environ.get("GITHUB_ACTIVITY_DB_SERVER", "localhost")
        database = os.environ.get("GITHUB_ACTIVITY_DB_NAME", "MultiAgentPlatform")
        driver = os.environ.get("GITHUB_ACTIVITY_DB_DRIVER", "ODBC Driver 18 for SQL Server")
        _require_setting("GITHUB_ACTIVITY_DB_SERVER", server)
        _require_setting("GITHUB_ACTIVITY_DB_DRIVER", driver)
        trust_certificate = _environment_bool(
            "GITHUB_ACTIVITY_DB_TRUST_SERVER_CERTIFICATE",
            default=True,
        )

        if database_url:
            parsed_url = make_url(database_url)
            if parsed_url.drivername != "mssql+pyodbc":
                raise ValueError("GITHUB_ACTIVITY_DATABASE_URL must use mssql+pyodbc.")
            _require_activity_database(_database_from_url(parsed_url))
        _require_activity_database(database)

        return cls(
            server=server,
            database=database,
            driver=driver,
            trust_server_certificate=trust_certificate,
            database_url=database_url,
        )

    def to_sqlalchemy_url(self) -> URL:
        """Build an ODBC URL using Windows Authentication and no credentials."""
        _require_activity_database(self.database)
        _require_setting("GITHUB_ACTIVITY_DB_SERVER", self.server)
        _require_setting("GITHUB_ACTIVITY_DB_DRIVER", self.driver)
        if self.database_url:
            url = make_url(self.database_url)
            _require_activity_database(_database_from_url(url))
            return url

        def odbc_value(value: str) -> str:
            return "{" + value.replace("}", "}}") + "}"

        trust_value = "yes" if self.trust_server_certificate else "no"
        connection_string = ";".join(
            (
                f"DRIVER={odbc_value(self.driver)}",
                f"SERVER={odbc_value(self.server)}",
                f"DATABASE={odbc_value(self.database)}",
                "Trusted_Connection=yes",
                "Encrypt=yes",
                f"TrustServerCertificate={trust_value}",
            )
        )
        return URL.create(
            "mssql+pyodbc",
            query={"odbc_connect": connection_string},
        )


def create_database_engine(settings: DatabaseSettings | None = None) -> Engine:
    """Create a quiet pooled engine for only the configured activity database."""
    configured = settings or DatabaseSettings.from_environment()
    _require_activity_database(configured.database)
    url = configured.to_sqlalchemy_url()
    _require_activity_database(_database_from_url(url))
    return create_engine(url, pool_pre_ping=True, echo=False)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create the reusable SQLAlchemy session factory for a database engine."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def initialize_activity_schema(engine: Engine) -> None:
    """Create the table and safely add the M7 significance field if needed."""
    if engine.dialect.name == "mssql":
        _require_activity_database(_database_from_url(engine.url))
    ActivityBase.metadata.create_all(
        bind=engine,
        tables=[GitHubActivityModel.__table__],
    )
    if engine.dialect.name == "mssql":
        columns = {column["name"] for column in inspect(engine).get_columns("github_activities")}
        if "significance" not in columns:
            # Additive compatibility change: existing rows receive the legacy-safe value.
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE [github_activities] ADD [significance] "
                        "NVARCHAR(16) NOT NULL CONSTRAINT "
                        "[DF_github_activities_significance] DEFAULT 'low' WITH VALUES"
                    )
                )
        for index in GitHubActivityModel.__table__.indexes:
            if index.name == "ix_github_activities_significance":
                index.create(bind=engine, checkfirst=True)
