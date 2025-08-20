
# Setup local persistence
1. Create .env file with the following content:
```.env
POSTGRES_HOST="localhost:5432"
POSTGRES_USER=<USERNAME>
POSTGRES_PASSWORD=<PASSWORD>
POSTGRES_DATABASE=kdcube
DEBUG=
```

2. Run the [persistence compose file](../docker-compose.yaml):
```
docker compose -f docker-compose.yaml --env-file .env up
```

3. Follow instructions from [DB-OPS-README.md](DB-OPS-README.md) to connect to the database and setup your account.
4. Other env in .env:
```.env
OPENAI_API_KEY=
HUGGING_FACE_KEY=
```


# RDS
Using SSH tunnel with port forwarding

```bash
ssh -i ~/.ssh/id_rsa_nestlogic -L 5432:kdcube.chnsfs2wwxtr.eu-west-1.rds.amazonaws.com:5432 ubuntu@34.250.63.191 -N -f
```
