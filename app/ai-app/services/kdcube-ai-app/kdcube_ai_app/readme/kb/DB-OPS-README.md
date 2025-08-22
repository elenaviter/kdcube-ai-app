Connect to PostgreSQL

In the scripts below, replace `<POSTGRES_USER>` with your desired username and `<YOUR PASSWORD>` with a secure password of your choice.

# PostgreSQL should have pgvector installed

Images:
`pgvector/pgvector:pg16`

```shell
docker exec -it ef-postgres psql -U <POSTGRES_USER> -d postgres
```

List the users and dbs
```shell
postgres=# \du
                             List of roles
 Role name |                         Attributes                         
-----------+------------------------------------------------------------
 ltuser    | Superuser, Create role, Create DB, Replication, Bypass RLS

postgres=# \l
                                                   List of databases
   Name    | Owner  | Encoding | Locale Provider |  Collate   |   Ctype    | ICU Locale | ICU Rules | Access privileges 
-----------+--------+----------+-----------------+------------+------------+------------+-----------+-------------------
 langtrace | ltuser | UTF8     | libc            | en_US.utf8 | en_US.utf8 |            |           | 
 postgres  | ltuser | UTF8     | libc            | en_US.utf8 | en_US.utf8 |            |           | 
 template0 | ltuser | UTF8     | libc            | en_US.utf8 | en_US.utf8 |            |           | =c/ltuser        +
           |        |          |                 |            |            |            |           | ltuser=CTc/ltuser
 template1 | ltuser | UTF8     | libc            | en_US.utf8 | en_US.utf8 |            |           | =c/ltuser        +
           |        |          |                 |            |            |            |           | ltuser=CTc/ltuser
(4 rows)
```

Create database
```shell
CREATE DATABASE kdcube;
```

Create another user and grant privileges
```shell
CREATE ROLE <POSTGRES_USER> WITH SUPERUSER LOGIN PASSWORD '<YOUR PASSWORD>';
ALTER ROLE <POSTGRES_USER> WITH SUPERUSER CREATEDB CREATEROLE REPLICATION BYPASSRLS;
GRANT ALL PRIVILEGES ON DATABASE kdcube TO <POSTGRES_USER>;
```

Relogin with new user
```shell
docker exec -it ef-postgres psql -U <POSTGRES_USER> -d kdcube
```

Log and check the extensions
```shell
SELECT extname FROM pg_extension;
```

Ensure vector extension is installed
```shell
CREATE EXTENSION IF NOT EXISTS vector;
```