# Service ECS Env (AWS Runtime)

Common AWS env vars used by services that export to CloudWatch or access AWS APIs.

## Required (all AWS)
| Variable             | Purpose                                     | Default | Scope         |
|----------------------|---------------------------------------------|---------|---------------|
| `AWS_REGION`         | AWS region for SDK calls                    | —       | all services  |
| `AWS_DEFAULT_REGION` | Alternate region variable recognized by SDK | —       | all services  |

## When running on ECS/EC2 with IAM role (recommended)
| Variable                      | Purpose                                                        | Default                               | Scope |
|-------------------------------|----------------------------------------------------------------|---------------------------------------|---|
| `AWS_EC2_METADATA_DISABLED`   | Must be `false` (or unset) so SDK can use instance/Task role   | `false`                               | all services |
| `NO_PROXY`                    | Ensure IMDS is reachable if proxy is used                      | `169.254.169.254,localhost,127.0.0.1` | all services |

## When running locally / outside AWS
Use one of these credential sources:

**Option A: environment keys**
| Variable                | Purpose                               |
|-------------------------|---------------------------------------|
| `AWS_ACCESS_KEY_ID`     | Access key                            |
| `AWS_SECRET_ACCESS_KEY` | Secret key                            |
| `AWS_SESSION_TOKEN`     | Session token (if using temporary creds) |

**Option B: shared config/profile**
| Variable | Purpose |
|---|---|
| `AWS_SDK_LOAD_CONFIG` | Enable shared config parsing (`1`) |
| `AWS_PROFILE` | Named profile in `~/.aws/config` |

## Notes
- For CloudWatch export, services must have `cloudwatch:PutMetricData` permissions.
- Prefer IAM roles in ECS/EC2 instead of static keys.
