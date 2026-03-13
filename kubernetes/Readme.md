# Deployment

k3d cluster create k3s-dev \
  --registry-create k3d-registry.localhost:5000 \


image:
  repository: k3d-registry.localhost:5000/myapp
  tag: dev

helm registry login k3d-registry.localhost:5000

helm package mychart
helm push mychart-0.1.0.tgz oci://k3d-registry.localhost:5000/charts
helm install myapp \
  oci://k3d-registry.localhost:5000/charts/mychart \
  --version 0.1.0

k3d-registry.localhost:5000/
├── myapp              (Docker images)
└── charts/mychart     (Helm charts)


===

! create physical folders

  HOST_KDCUBE_STORAGE_PATH: /Users/viacheslav/work/NestLogic/kdcube/data/kdcube-storage
  HOST_BUNDLES_PATH: /Users/viacheslav/work/NestLogic/kdcube/data/bundles
  HOST_EXEC_WORKSPACE_PATH: /Users/viacheslav/work/NestLogic/kdcube/data/exec-workspace


from app/ai-app dir


docker build -t k3d-registry.localhost:5000/kdcube-chat:latest .
docker push k3d-registry.localhost:5000/kdcube-chat:latest

docker build -f deployment/docker/all_in_one/Dockerfile_PostgresSetup -t kdcube-postgres-setup:latest .
docker tag kdcube-postgres-setup:latest k3d-registry.localhost:5000/kdcube-postgres-setup:latest
docker push k3d-registry.localhost:5000/kdcube-postgres-setup:latest

docker build -f deployment/docker/all_in_one/Dockerfile_Chat -t kdcube-chat:latest .
docker tag kdcube-chat:latest k3d-registry.localhost:5000/kdcube-chat:latest
docker push k3d-registry.localhost:5000/kdcube-chat:latest

docker build -f deployment/docker/all_in_one/Dockerfile_UI \
--build-arg UI_SOURCE_PATH=ui/chat-web-app \
--build-arg UI_ENV_BUILD_RELATIVE=deployment/docker/all_in_one/sample_env/.env.ui.build \
--build-arg PATH_TO_FRONTEND_CONFIG_JSON=ui/chat-web-app/public/config.hardcoded.json \
--build-arg NGINX_UI_CONFIG_FILE_PATH=deployment/docker/all_in_one/nginx_ui.conf \
-t kdcube-web-ui:latest .
docker tag kdcube-web-ui:latest k3d-registry.localhost:5000/kdcube-web-ui:latest
docker push k3d-registry.localhost:5000/kdcube-web-ui:latest


docker build -f deployment/docker/all_in_one/Dockerfile_Exec -t kdcube-executor:latest .
docker tag kdcube-executor:latest k3d-registry.localhost:5000/kdcube-executor:latest
docker push k3d-registry.localhost:5000/kdcube-executor:latest

kubectl -n kdcube-ai-app port-forward svc/postgres-db-rw 5432:5432

kubectl create ns kdcube-ai-app

helm install postgres-db ./postgres-db --namespace kdcube-ai-app

helm install redis ./redis --namespace kdcube-ai-app

helm upgrade --install clamav oci://registry.gitlab.com/xrow-public/helm-clamav/charts/clamav \
  --version 1.8.23 \
  --namespace kdcube-ai-app

helm upgrade -i kdcube-web-ui ./kubernetes/kdcube-web-ui --namespace kdcube-ai-app