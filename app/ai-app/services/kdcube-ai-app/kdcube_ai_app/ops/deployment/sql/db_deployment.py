# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# ops/deployment/sql/db_deployment.py

import argparse
import os, sys, re

# Add the directory to sys.path if it's not already there
from kdcube_ai_app.infra.relational.psql.psql_base import PostgreSqlDbMgr

SYSTEM_COMPONENT = "kdcube-system-schema"
PROJECT_COMPONENT = "kdcube-proj-schema"

PROJECT_DEFAULT_SCHEMA = "kdcube_default"
SYSTEM_SCHEMA = "kdcubesystem"

SUPPORTED_COMPONENTS = ["experts-and-registries", "rbac", "knowledge_base",
                        "event-log", SYSTEM_COMPONENT]

sql_location = os.path.dirname(__file__)



def safe_schema_name(name: str) -> str:
    """
    Turn an arbitrary string into a safe PostgreSQL schema name:
      - lowercase
      - only letters, digits, and underscores
      - starts with a letter or underscore
      - maximum length of 63 characters
      - fallback to '_schema' if name is empty after sanitization
    """
    # 1) Replace any character not a letter, digit, or underscore with '_'
    sanitized = re.sub(r'[^A-Za-z0-9_]', '_', name)
    # 2) Collapse multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # 3) Strip leading/trailing underscores
    sanitized = sanitized.strip('_')
    # 4) Lowercase
    sanitized = sanitized.lower()
    # 5) Ensure it starts with a letter or underscore
    if not re.match(r'^[a-z_]', sanitized):
        sanitized = '_' + sanitized
    # 6) Truncate to PostgreSQL’s max identifier length (63 chars)
    max_len = 63
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len]
    # 7) Fallback if empty
    return sanitized or '_schema'

def run(op, component, tenant = None, project = None, app=None):
    mgr = PostgreSqlDbMgr()
    if not component:
        error = "Please specify a component."
        print(error)
        raise Exception(error)

    schema_name = safe_schema_name(project or "default-project")
    if tenant:
        substitutions = { "SCHEMA": f"kdcube_{tenant}_{schema_name}", "SYSTEM_SCHEMA": SYSTEM_SCHEMA }
    else:
        substitutions = { "SCHEMA": f"kdcube_{schema_name}", "SYSTEM_SCHEMA": SYSTEM_SCHEMA }
    if op == "deploy":
        if app:
            schema_file = os.path.join(sql_location, app, f"deploy-{component}.sql")
        else:
            schema_file = os.path.join(sql_location, f"deploy-{component}.sql")
        try:
            mgr.execute_sql_file(schema_file, substitutions=substitutions)
            print("Schema deployed successfully.")
        except Exception as e:
            error = f"Error deploying schema: {e}"
            print(error)
            raise e

    elif op == "delete":
        if app:
            delete_file = os.path.join(sql_location, app, f"drop-{component}.sql")
        else:
            delete_file = os.path.join(sql_location, f"drop-{component}.sql")
        try:
            mgr.execute_sql_file(delete_file, substitutions=substitutions)
            print("Schema deleted successfully.")
        except Exception as e:
            error = f"Error deleting schema: {e}"
            print(error)
            raise e
    else:
        print("Please specify --deploy or --delete.")

def main(args, parser):
    """
    Main function to handle CLI arguments and execute scripts.
    """
    if args.deploy:
        op = "deploy"
    elif args.delete:
        op = "delete"
    else:
        print("Please specify --deploy or --delete.")
        parser and parser.print_help()
        return False
    component = args.component

    if not component:
        print("Please specify a component.")
        parser and parser.print_help()
        sys.exit(1)

    project = args.project
    if not project and component != SYSTEM_COMPONENT:
        print("Please specify a project name.")
        parser and parser.print_help()
        sys.exit(1)

    tenant = args.tenant
    if not tenant and component != SYSTEM_COMPONENT:
        print("Please specify a tenant name.")
        parser and parser.print_help()
        sys.exit(1)
    app = args.app

    run(op, component, tenant, project, app)

if __name__ == "__main__":
    # def load_env():
    #     _ = load_dotenv(find_dotenv(".env.prod"))
    # load_env()

    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    parser = None

    # debug = os.environ["DEBUG"] == "true"
    debug = True
    # debug = True
    # deploy = True
    # deploy = False
    # delete = True

    delete = False
    deploy = True
    # deploy = True
    # delete = False

    component = "experts-and-registries"
    component = PROJECT_COMPONENT
    tenant = "home"

    # component = SYSTEM_COMPONENT

    sync_pricing = True
    class MArgs:
        def __init__(self):
            self.delete = delete
            self.deploy = deploy
            self.sync_pricing = sync_pricing
            self.component = component
            self.project = "running_shoes"
            self.app = "knowledge_base"
            self.tenant = "home"

    if debug:
        args = MArgs()
    else:
        parser = argparse.ArgumentParser(description="Database Tool for KDCube schema deployments.")
        parser.add_argument(
            "--component", action="store", choices=SUPPORTED_COMPONENTS,
            help=f"Name of the component to deploy {SUPPORTED_COMPONENTS}."
        )
        parser.add_argument(
            "--deploy", action="store_true", help="Deploy the database schema and indices."
        )
        parser.add_argument(
            "--delete", action="store_true", help="Delete the database schema and indices."
        )
        parser.add_argument(
            "--project", help="Name of the project (name of the schema)"
        )
        parser.add_argument(
            "--tenant", help="Name of the tenant"
        )
        parser.add_argument(
            "--app", help="Name of the app (namespace of the deployment scripts)"
        )
        args = parser.parse_args()
    main(args, parser)
