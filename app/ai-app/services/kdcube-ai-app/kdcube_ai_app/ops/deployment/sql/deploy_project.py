from kdcube_ai_app.ops.deployment.sql.db_deployment import run as provision, SYSTEM_COMPONENT, SYSTEM_SCHEMA, PROJECT_COMPONENT


def step_provision(tenant, project):
    # provision("deploy", SYSTEM_COMPONENT)
    provision("deploy", PROJECT_COMPONENT, tenant, project.replace("-", "_"), "knowledge_base")

def step_deprovision(tenant, project):
    provision("delete", PROJECT_COMPONENT, tenant, project.replace("-", "_"), "knowledge_base")

if __name__ == "__main__":
    # def load_env():
    #     _ = load_dotenv(find_dotenv(".env.prod"))
    # load_env()
    # generate_datasource_rn
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    import os
    project = os.environ.get("DEFAULT_PROJECT_NAME", None)
    tenant = os.environ.get("DEFAULT_TENANT", None)
    step_provision(tenant, project)
    # step_deprovision(tenant, project)