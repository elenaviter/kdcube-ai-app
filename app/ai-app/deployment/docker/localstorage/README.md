```shell
mkdir -p ./data/{postgres,redis,clamav-db,neo4j/{data,logs,plugins,import}}
```

```shell
chmod -R 0777 data 
```

```shell
docker compose up -d
```