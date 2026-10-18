# Deploying to GCP

Deployed this way on 2026-10-18 (30 resources in one project, us-central1). Needs a GCP project with billing,
which only its owner can create. Cloud Run's free tier covers 2 million requests a month; the budget alert in
Terraform defaults to $5.

1. Log in, pick the project and create a bucket for Terraform state:

   ```bash
   gcloud auth login && gcloud auth application-default login
   gcloud config set project <project-id>
   gcloud auth application-default set-quota-project <project-id>
   gcloud storage buckets create gs://<project-id>-tfstate --location us-central1 --uniform-bucket-level-access
   terraform -chdir=infra/terraform init -backend-config="bucket=<project-id>-tfstate"
   ```

2. Create what the service needs before it can start (registry, models bucket, API-key secret):

   ```bash
   VARS="-var project_id=<project-id> -var billing_account_id=<billing-account-id> -var alert_email=<you@example.com>"
   terraform -chdir=infra/terraform apply $VARS -target=google_artifact_registry_repository.images \
     -target=google_storage_bucket.models -target=google_secret_manager_secret_iam_member.service_reads_keys
   ```

3. Store the API keys (client:key pairs, keys of 16+ characters) as the secret's first version, upload the model
   and push the image:

   ```bash
   printf 'viewer:%s,loadtest:%s' "$(openssl rand -hex 16)" "$(openssl rand -hex 16)" > ~/.config/pathology/api-keys
   gcloud secrets versions add pathology-api-keys --data-file="$HOME/.config/pathology/api-keys"
   gcloud storage cp -r models/pcam gs://<project-id>-pathology-models/
   gcloud auth configure-docker us-central1-docker.pkg.dev
   docker build -f docker/service.Dockerfile -t us-central1-docker.pkg.dev/<project-id>/pathology/service:latest .
   docker push us-central1-docker.pkg.dev/<project-id>/pathology/service:latest
   ```

4. Apply the rest (Cloud Run, Workload Identity Federation for GitHub Actions, budget and alert policies):

   ```bash
   terraform -chdir=infra/terraform apply $VARS
   ```

   The budget uses a second provider with `user_project_override`: the Budgets API rejects user credentials that
   carry no quota project, and the first apply failed there before this was added.

5. Check it and measure it (results in `reports/cloud-run/`):

   ```bash
   URL=$(terraform -chdir=infra/terraform output -raw service_url)
   python scripts/cloud_smoke.py "$URL" --key-file ~/.config/pathology/api-keys --label warm
   docker run --rm -v "$PWD/bench/k6:/scripts:ro" -v "$PWD/service/tests/fixtures:/fixtures:ro" \
     -v "$PWD/reports/cloud-run/k6:/out" -e FIXTURES_DIR=/fixtures -e TARGET="$URL" -e API_KEY=<loadtest key> \
     -e LABEL=cloudrun -e RATE=50 -e PATCHES=1 -e DURATION=30s grafana/k6:1.3.0 run /scripts/classify.js
   ```

   For a cold start, wait until Cloud Monitoring shows no instances (`run.googleapis.com/container/instance_count`,
   active and idle both 0; about 15 minutes after the last request), then run the smoke test with `--label cold`.

6. Point the viewer at it: build `web/viewer` with `VITE_SERVICE_URL=$URL` and the viewer's key, and host
   `web/viewer/dist` on any static host. Add that origin to `CORS_ORIGINS`. (Not done yet.)
