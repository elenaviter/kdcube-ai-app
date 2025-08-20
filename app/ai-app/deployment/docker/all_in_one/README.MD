# All-In-One Setup

## First run
To initialize Postgres run this once
```shell
  docker compose run postgres-setup
```


### Create local KB storage if does not exist (optional)
```shell
  mkdir "kb-storage" -p && chmod 777 "kb-storage"
```

### Create new clean local KB storage (optional)
```shell
  rm -rf "kb-storage" && mkdir "kb-storage" -p && chmod 777 "kb-storage"
```

## Run


### Backend
```shell
  docker compose --profile "backend" up -d --remove-orphans
```
With image rebuild
```shell
  docker compose --profile "backend" up -d --remove-orphans --build
```

### Frontend
```shell
  docker compose --profile "frontend" up -d --remove-orphans
```
With image rebuild
```shell
  docker compose --profile "frontend" up -d --remove-orphans --build
```
## Quick commands

### Stop and remove backend services
```shell
  docker compose stop chat kb dramatiq && docker compose rm chat kb dramatiq -f
```

### Chat Rebuild
```shell
  docker compose stop chat && docker compose rm chat -f && docker compose up chat -d --build 
```

### KB Rebuild
```shell
  docker compose stop kb && docker compose rm kb -f && docker compose up kb -d --build 
```

### Dramatiq Rebuild
```shell
  docker compose stop dramatiq && docker compose rm dramatiq -f && docker compose up dramatiq -d --build 
```

### UI Rebuild
```shell
  docker compose stop web-ui && docker compose rm web-ui -f && docker compose up web-ui -d --build 
```

### Proxy Rebuild
```shell
  docker compose stop web-proxy && docker compose rm web-proxy -f && docker compose up web-proxy -d --build 
```