# Deployment Guide

This project supports an AWS-first prototype deployment using temporary AWS URLs. You can attach a custom domain later without rebuilding the app.

## Architecture

- **Frontend:** AWS Amplify Hosting (Vite build from `frontend/`)
- **API:** ECS Fargate behind an Application Load Balancer
- **Auth:** Amazon Cognito User Pool
- **Chat history:** DynamoDB table keyed by `user_id` + `conversation_id`
- **Retrieval:** Pinecone + local JSON index loaded from S3/EFS at startup

## 1. Frontend on Amplify

1. Connect the GitHub repo to Amplify Hosting.
2. Amplify reads [`amplify.yml`](amplify.yml) and builds the frontend from `frontend/`.
3. Set Amplify environment variables:
   - `VITE_API_BASE_URL=https://<your-alb-dns-name>`
4. Use the default Amplify URL first, e.g. `https://main.xxxxx.amplifyapp.com`.
5. Later, add a custom domain in Amplify without changing app code.

## 2. API on ECS Fargate

1. Create an ECR repository: `gsi-standards-rag-api`
2. Create an ECS cluster: `gsi-standards-rag`
3. Create an ALB + target group pointing to port `8000`
4. Use the ALB DNS name as the temporary API URL
5. Store secrets in AWS Secrets Manager / SSM Parameter Store
6. Update placeholders in [`deploy/ecs-task-definition.json`](deploy/ecs-task-definition.json)
7. Configure GitHub Actions secret `AWS_DEPLOY_ROLE_ARN`
8. Push to `main` to trigger [`.github/workflows/deploy-api.yml`](.github/workflows/deploy-api.yml)

## 3. Cognito

Create a Cognito User Pool with:

- Email sign-in
- App client without client secret (public SPA/client flow)
- `USER_PASSWORD_AUTH` enabled if using `/auth/login`

Set these env vars on the API:

```env
AUTH_REQUIRED=1
COGNITO_USER_POOL_ID=...
COGNITO_APP_CLIENT_ID=...
COGNITO_REGION=us-east-1
```

## 4. DynamoDB chat history

Create a table:

- Name: `gsi-conversations`
- Partition key: `user_id` (String)
- Sort key: `conversation_id` (String)

Set:

```env
DYNAMODB_CONVERSATIONS_TABLE=gsi-conversations
CONVERSATION_TTL_DAYS=90
```

When unset locally, the API uses in-memory chat storage.

## 5. CORS

Set allowed frontend origins on the API:

```env
CORS_ALLOW_ORIGINS=https://main.xxxxx.amplifyapp.com
```

## 6. Domain later

You do not need to buy a domain for the first demo.

1. Keep Amplify default URL for the UI
2. Keep ALB DNS for the API
3. Later:
   - Add ACM certificate
   - Map Route 53 record to Amplify
   - Map API subdomain to ALB

## Local development

```bash
pip install -e ".[api,pdf,pinecone,llm,auth,aws,dev]"
uvicorn standards_rag.api:app --reload
cd frontend && npm install && npm run dev
```

Auth is optional locally (`AUTH_REQUIRED=0`).
