# The JavaScript viewer, built by Vite and served by nginx.
FROM node:24-bookworm-slim AS build
WORKDIR /web
COPY web/package.json web/package-lock.json ./
COPY web/client/package.json client/
COPY web/viewer/package.json viewer/
RUN npm ci --no-audit --no-fund
COPY web/client client
COPY web/viewer viewer
ARG VITE_SERVICE_URL=http://localhost:8080
ARG VITE_API_KEY=local-dev-key-viewer-000001
ENV VITE_SERVICE_URL=$VITE_SERVICE_URL VITE_API_KEY=$VITE_API_KEY
RUN npm run build -w viewer

FROM nginxinc/nginx-unprivileged:1.29-alpine
COPY --from=build /web/viewer/dist /usr/share/nginx/html
EXPOSE 8080
