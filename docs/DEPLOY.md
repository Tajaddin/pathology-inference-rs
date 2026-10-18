# Deploying to GCP (not done yet)

Needs a GCP project with billing, which only its owner can create. Cloud Run's free tier covers 2 million
requests a month; the budget alert in Terraform defaults to $5.

1. Create the project and a bucket for Terraform state:

   ```bash
   gcloud projects create <project-id> && gcloud config set project <project-id>
   gcloud billing projects link <project-id> --billing-account <billing-account-id>
   gcloud storage buckets create gs://<project-id>-tfstate --location us-central1
   ```

2. Apply the infrastructure (Artifact Registry, models bucket, Secret Manager, Cloud Run, Workload Identity
   Federation for GitHub Actions, budget and alert policies):

   ```bash
   terraform -chdir=infra/terraform init -backend-config="bucket=<project-id>-tfstate"
   terraform -chdir=infra/terraform apply -var project_id=<project-id> \
     -var billing_account_id=<billing-account-id> -var alert_email=<you@example.com>
   ```

   The first apply creates the service before an image exists; it becomes ready after step 4.
3. Store the API keys (client:key pairs, keys of 16+ characters) as the secret's first version:

   ```bash
   printf 'viewer:%s,loadtest:%s' "$(openssl rand -hex 16)" "$(openssl rand -hex 16)" |
     gcloud secrets versions add pathology-api-keys --data-file=-
   ```

4. Upload the model and push the image:

   ```bash
   gcloud storage cp -r models/pcam gs://<project-id>-pathology-models/
   gcloud auth configure-docker us-central1-docker.pkg.dev
   docker build -f docker/service.Dockerfile -t "$(terraform -chdir=infra/terraform output -raw image)" .
   docker push "$(terraform -chdir=infra/terraform output -raw image)"
   gcloud run services update pathology-service --region us-central1 --image "$(terraform -chdir=infra/terraform output -raw image)"
   ```

5. Point the viewer at it: build `web/viewer` with `VITE_SERVICE_URL=$(terraform output -raw service_url)` and
   the viewer's key, and host `web/viewer/dist` on any static host. Add that origin to `CORS_ORIGINS`.
6. Re-run the k6 script against the Cloud Run URL to get cold-start and steady-state numbers for the README;
   the local numbers say nothing about Cloud Run.
