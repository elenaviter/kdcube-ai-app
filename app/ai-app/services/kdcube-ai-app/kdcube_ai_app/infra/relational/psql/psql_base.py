import os
import psycopg2
from psycopg2.extras import execute_values

from typing import Union, List, Optional, Dict


class PostgreSqlDbMgr:
    def __init__(self, connection_params: Optional[Dict[str, str]] = None):
        if not connection_params:
            connection_params = {}
        self.host = connection_params.get("host") or os.environ.get("POSTGRES_HOST")
        self.port = connection_params.get("port") or os.environ.get("POSTGRES_PORT")
        self.database = connection_params.get("database") or os.environ.get("POSTGRES_DATABASE")

        self.username = connection_params.get("username") or os.environ.get("POSTGRES_USER")
        self.password = connection_params.get("password") or os.environ.get("POSTGRES_PASSWORD")

        self.ssl = os.environ.get("POSTGRES_SSL", "false").lower() == "true"
        if self.port:
            self.database_url = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            self.database_url = f"postgresql://{self.username}:{self.password}@{self.host}/{self.database}"

        if self.ssl:
            self.database_url += "?sslmode=require"

    def get_connection(self):
        return psycopg2.connect(self.database_url)

    def execute_sql_string(self, sql: str):
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                conn.commit()
        print(f"Executed SQL: {sql}")

    def execute_sql_file(self, file_path, substitutions=None):
        """
        Execute a SQL file.
        """
        with open(file_path, 'r') as file:
            sql = file.read()
            if substitutions:
                for key, value in substitutions.items():
                    if value is not None:
                        sql = sql.replace(f"<{key}>", value)
            self.execute_sql_string(sql)
        print(f"Executed SQL file: {file_path}")

    def execute_sql(
            self,
            sql: str,
            data: Union[tuple, List[tuple]] = None,
            as_dict: bool = True,
            debug: bool = False,
            bulk: bool = False
    ):
        """
        Execute arbitrary SQL with optional data.

        :param sql: The SQL query to execute.
        :param data: Optional tuple or list-of-tuples of data to bind.
        :param as_dict: Whether to return the results as a dictionary (list[dict])
                        or as a dict with "columns" / "rows".
        :param debug: If True, prints debug info.
        :param bulk: Use execute_values for bulk insertion.
        :return: Query results if it's a SELECT or if there's a RETURNING clause,
                 otherwise None.
        """
        if debug:
            print(f"Executing SQL: {sql}")
            print(f"With parameters: {data}")


        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # If we want to use execute_values for bulk insertion
                if bulk and data:
                    execute_values(cur, sql, data)
                elif data:
                    cur.execute(sql, data)
                else:
                    cur.execute(sql)

                # 1) Check for "SELECT" or "RETURNING" in the query
                #    If present, we fetch rows

                stripped = sql.lstrip().lower()
                is_select = stripped.startswith("select") or stripped.startswith("with")
                has_returning = "returning" in stripped

                if is_select or has_returning:
                    # 2) fetch rows
                    rows = cur.fetchall()
                    columns = [desc[0] for desc in cur.description]
                    if as_dict:
                        return [dict(zip(columns, row)) for row in rows]
                    else:
                        return {"columns": columns, "rows": rows}

                # Otherwise (no data to return, e.g. normal INSERT/UPDATE/DELETE w/o returning)
                conn.commit()
                return None

    def execute_sql_(
            self,
            sql: str,
            data: Union[tuple, List[tuple]] = None,
            as_dict: bool = True,
            debug: bool = False,
            bulk: bool = False,
            conn=None
    ):
        """
        Execute arbitrary SQL with optional data.

        :param sql: The SQL query to execute.
        :param data: Optional tuple or list-of-tuples of data to bind.
        :param as_dict: Whether to return the results as a dictionary (list[dict])
                        or as a dict with "columns" / "rows".
        :param debug: If True, prints debug info.
        :param bulk: Use execute_values for bulk insertion.
        :param conn: Optional existing database connection to use instead of creating a new one.
        :return: Query results if it's a SELECT or if there's a RETURNING clause,
                 otherwise None.
        """
        if debug:
            print(f"Executing SQL: {sql}")
            print(f"With parameters: {data}")

        # Determine if we should manage connection (create/close) or use the provided one
        should_manage_connection = conn is None

        try:
            # Create connection only if one wasn't provided
            if should_manage_connection:
                conn = self.get_connection()

            with conn.cursor() as cur:
                # If we want to use execute_values for bulk insertion
                if bulk and data:
                    execute_values(cur, sql, data)
                elif data:
                    cur.execute(sql, data)
                else:
                    cur.execute(sql)

                # 1) Check for "SELECT" or "RETURNING" in the query
                #    If present, we fetch rows
                lower_sql = sql.strip().lower()
                is_select = lower_sql.startswith("select")
                has_returning = "returning" in lower_sql

                if is_select or has_returning:
                    # 2) fetch rows
                    rows = cur.fetchall()
                    columns = [desc[0] for desc in cur.description]
                    if as_dict:
                        return [dict(zip(columns, row)) for row in rows]
                    else:
                        return {"columns": columns, "rows": rows}

                # Otherwise (no data to return, e.g. normal INSERT/UPDATE/DELETE w/o returning)
                if should_manage_connection:
                    conn.commit()
                return None
        finally:
            # Only close the connection if we created it
            if should_manage_connection and conn is not None:
                conn.close()

    def list_schemas(self) -> list:
        sql = "SELECT schema_name FROM information_schema.schemata;"
        # Execute the query using the existing method
        results = self.execute_sql(sql)
        # Extract and return the schema names
        return [row["schema_name"] for row in results] if results else []