VENV_NAME:=venv



ENV_FILE := .env

ifneq (,$(wildcard $(ENV_FILE)))
  include $(ENV_FILE)
  export $(shell grep -v '^#\|^$$' $(ENV_FILE) | sed 's/ *=.*//' | xargs)
#   export $(shell sed 's/=.*//' $(ENV_FILE))
endif

# Variables
REGISTRY_NAME?=

# SERVICE_DIR ?=
IMAGE_NAME ?=
IMAGE_TAG ?= $(shell git rev-parse --short HEAD)

# DOCKER_IMAGE_NAME:=${REGISTRY_NAME}/$(IMAGE_NAME)

K8S_NAMESPACE=farud-api-dev
K8S_CONFIG_PATH=k8s/overlays/dev

FRAUD_DETECTION_API=fraud-detection-api
FRAUD_DETECTION_FRONTEND=fraud-detection-frontend

DOCKER_ADDITIONAL_ARGS=

activate_env:
	source $(VENV_NAME)/bin/activate

setup_mldev:
	@pip install -r notebook/requirements.txt
	@python -m ipykernel install --user --name=$(VENV_NAME) --display-name "Python (fraud-venv)"

download_data:
	@pip install -r notebook/requirements.txt
	@cd notebook && \
	python download_data.py



# Docker build and push targets
.PHONY: build
build: DOCKER_IMAGE_NAME=$(REGISTRY_NAME)/$(IMAGE_NAME)
build:	
	@echo "Building Docker image.."
	@echo "IMAGE:: $(DOCKER_IMAGE_NAME):$(IMAGE_TAG)"
	@cd $(SERVICE_DIR) && \
	docker build -t $(DOCKER_IMAGE_NAME):$(IMAGE_TAG) .
	docker tag $(DOCKER_IMAGE_NAME):$(IMAGE_TAG) $(DOCKER_IMAGE_NAME):latest

# Run the application container
.PHONY: run
run: DOCKER_IMAGE_NAME=$(REGISTRY_NAME)/$(IMAGE_NAME)
run:
	@echo "Running Docker image.."
	@echo "IMAGE::" $(DOCKER_IMAGE_NAME):$(IMAGE_TAG)
	docker run --rm -it -p ${PORT}:${PORT} ${DOCKER_ADDITIONAL_ARGS} $(DOCKER_IMAGE_NAME):latest

.PHONY: push
push: DOCKER_IMAGE_NAME=$(REGISTRY_NAME)/$(IMAGE_NAME)
push:
	@echo "Pushing Docker image to registry..."
	docker push $(DOCKER_IMAGE_NAME):$(IMAGE_TAG)

.PHONY: deploy
update_manifest: DOCKER_IMAGE_NAME=$(REGISTRY_NAME)/$(IMAGE_NAME)
update_manifest:
# echo "building and deploying $(SERVICE)..."
# echo image name :
# docker build -t image_name docker_build_args -f docker_file .
# docker tag image_name registry_name/fullanme
# docker push registry_name/fullanme
	@echo "updating manifests..."
	@echo $(MANIFESTS_DIR)/overlays/$(DEPLOYMENT_ENV) 
	@echo "$(IMAGE_NAME)=$(DOCKER_IMAGE_NAME):$(IMAGE_TAG)"
	@cd $(MANIFESTS_DIR)/overlays/$(DEPLOYMENT_ENV) && \
	kustomize edit set image "$(IMAGE_NAME)=$(DOCKER_IMAGE_NAME):$(IMAGE_TAG)"

	

# Kubernetes deployment targets
.PHONY: deploy
deploy:
	@echo "Applying Kubernetes manifests..."
	@echo $(MANIFESTS_DIR)/overlays/$(DEPLOYMENT_ENV) 
	kubectl apply -k $(MANIFESTS_DIR)/overlays/$(DEPLOYMENT_ENV) 

undeploy:
	@echo "Undeploying Kubernetes manifests..."
	@echo $(MANIFESTS_DIR)/overlays/$(DEPLOYMENT_ENV) 
	kubectl delete -k $(MANIFESTS_DIR)/overlays/$(DEPLOYMENT_ENV) 

# # Testing target (requires pytest and API server running)
# test:
# 	@echo "Running tests..."
# 	pytest tests/

# Clean up Docker images
clean: DOCKER_IMAGE_NAME=$(REGISTRY_NAME)/$(IMAGE_NAME)
clean:
	@echo "Removing Docker images..."
	docker rmi $(DOCKER_IMAGE_NAME):$(IMAGE_TAG)

# Build and deploy pipeline
build-and-deploy: build push deploy

# Build, test, and deploy pipeline
build-test-deploy: build push test deploy


## 


api_build: IMAGE_NAME=$(FRAUD_DETECTION_API)
api_build: SERVICE_DIR=services/fraud_detection/
api_build: build

api_run: IMAGE_NAME=$(FRAUD_DETECTION_API)
api_run: DOCKER_ADDITIONAL_ARGS="-eMODEL_FILE=models/best_model.pkl"
api_run: PORT=8000
api_run: run

api_push: IMAGE_NAME=$(FRAUD_DETECTION_API)
api_push: push


frontend_build: IMAGE_NAME=$(FRAUD_DETECTION_FRONTEND)
frontend_build: SERVICE_DIR=services/frontend/
frontend_build: build

frontend_run: IMAGE_NAME=$(FRAUD_DETECTION_FRONTEND)
frontend_run: PORT=8501
frontend_run: DOCKER_ADDITIONAL_ARGS="-eAPI_URL=http://localhost:8000"
frontend_run: run

frontend_push: IMAGE_NAME=$(FRAUD_DETECTION_FRONTEND)
frontend_push: push


api_update_manifest: MANIFESTS_DIR=manifest/api
api_update_manifest: IMAGE_NAME=$(FRAUD_DETECTION_API)
api_update_manifest: update_manifest

dev_api_update_manifest: DEPLOYMENT_ENV=dev
dev_api_update_manifest: api_update_manifest

prod_api_update_manifest: DEPLOYMENT_ENV=prod
prod_api_update_manifest: api_update_manifest

frontend_update_manifest: MANIFESTS_DIR=manifest/frontend
frontend_update_manifest: IMAGE_NAME=$(FRAUD_DETECTION_FRONTEND)
frontend_update_manifest: update_manifest

dev_frontend_update_manifest: DEPLOYMENT_ENV=dev
dev_frontend_update_manifest: frontend_update_manifest

prod_frontend_update_manifest: DEPLOYMENT_ENV=prod
prod_frontend_update_manifest: frontend_update_manifest


api_deploy: MANIFESTS_DIR=manifest/api
api_deploy: deploy

api_undeploy: MANIFESTS_DIR=manifest/api
api_undeploy: undeploy

dev_api_deploy: DEPLOYMENT_ENV=dev
dev_api_deploy: api_deploy

prod_api_deploy: DEPLOYMENT_ENV=prod
prod_api_deploy: api_deploy


dev_api_undeploy: DEPLOYMENT_ENV=dev
dev_api_undeploy: api_undeploy

prod_api_undeploy: DEPLOYMENT_ENV=prod
prod_api_undeploy: api_undeploy

dev_api_full_build: api_build api_push dev_api_update_manifest dev_api_deploy
prod_api_full_build: api_build api_push prod_api_update_manifest prod_api_deploy



frontend_deploy: MANIFESTS_DIR=manifest/frontend
frontend_deploy: deploy

frontend_undeploy: MANIFESTS_DIR=manifest/frontend
frontend_undeploy: undeploy

dev_frontend_deploy: DEPLOYMENT_ENV=dev
dev_frontend_deploy: frontend_deploy

prod_frontend_deploy: DEPLOYMENT_ENV=prod
prod_frontend_deploy: frontend_deploy

dev_frontend_undeploy: DEPLOYMENT_ENV=dev
dev_frontend_undeploy: frontend_undeploy
prod_frontend_undeploy: DEPLOYMENT_ENV=prod
prod_frontend_undeploy: frontend_undeploy

dev_frontend_full_build: frontend_build frontend_push dev_frontend_update_manifest dev_frontend_deploy
prod_frontend_full_build: frontend_build frontend_push prod_frontend_update_manifest prod_frontend_deploy

dev_deploy_all: 
	$(MAKE) dev_api_deploy 
	$(MAKE) dev_frontend_deploy

dev_undeploy_all: 
	$(MAKE) dev_api_undeploy
	$(MAKE) dev_frontend_undeploy


dev_deploy_all_full: 
	$(MAKE) dev_api_full_build 
	$(MAKE) dev_frontend_full_build

dev_api_integation_test:
	@echo ""
	@echo "**Note******"
	@echo "Make sure virtual env is active and pytest is installed"
	@echo ""
	@cd services/fraud_detection/integration/ && \
	pytest

kind-init:
	kind create cluster --config kind.config
	sleep 5
	kubectl apply -f https://kind.sigs.k8s.io/examples/ingress/deploy-ingress-nginx.yaml
	sleep 30
	kubectl wait --namespace ingress-nginx \
	--for=condition=ready pod \
	--selector=app.kubernetes.io/component=controller \
	--timeout=90s

kind-delete:
	kind delete cluster

# Help command to display available targets
help:
	@echo "Available Makefile targets:"
	@echo ""
	@echo "Service Build Targets:"
#	@echo "  build					- Build both API and frontend services."
	@echo "  api_build				- Build the API service."
	@echo "  frontend_build			- Build the frontend service."
	@echo ""
	@echo "Service Run Targets:"
#	@echo "  run					- Run both API and frontend services."
	@echo "  api_run				- Run the API service."
	@echo "  frontend_run				- Run the frontend service."
	@echo ""
	@echo "Service Push Targets:"
#	@echo "  push					- Push both API and frontend services."
	@echo "  api_push				- Push the API service."
	@echo "  frontend_push				- Push the frontend service."
	@echo ""
	@echo "Manifest Update Targets:"
#	@echo "  update_manifest			- Update manifests for both API and frontend services."
	@echo "  dev_api_update_manifest		- Update the dev manifest for the API service."
	@echo "  prod_api_update_manifest		- Update the prod manifest for the API service."
	@echo "  dev_frontend_update_manifest		- Update the dev manifest for the frontend service."
	@echo "  prod_frontend_update_manifest		- Update the prod manifest for the frontend service."
	@echo ""
	@echo "Service Deploy Targets: (deploy to kind cluster)"
#	@echo "  deploy						- Deploy both API and frontend services."
	@echo "  dev_api_deploy			- Deploy the dev API service."
	@echo "  prod_api_deploy			- Deploy the prod API service."
	@echo "  dev_frontend_deploy			- Deploy the dev frontend service."
	@echo "  prod_frontend_deploy			- Deploy the prod frontend service."
	@echo ""
	@echo "Service Deploy Targets: (deploy to kind cluster)"
#	@echo "  deploy						- Deploy both API and frontend services."
	@echo "  dev_api_undeploy			- Undeploy the dev API service."
	@echo "  prod_api_undeploy			- Undeploy the prod API service."
	@echo "  dev_frontend_undeploy			- Undeploy the dev frontend service."
	@echo "  prod_frontend_undeploy			- Undeploy the prod frontend service."
	@echo ""
	@echo "Service Clean Targets:"
	@echo "  clean					- Clean up resources for both API and frontend services."
	@echo ""
	@echo "Full Build Targets:"
	@echo "  dev_api_full_build			- Full build process for the dev API (build, push, update manifest, deploy)."
	@echo "  prod_api_full_build			- Full build process for the prod API (build, push, update manifest, deploy)."
	@echo "  dev_frontend_full_build		- Full build process for the dev frontend (build, push, update manifest, deploy)."
	@echo "  prod_frontend_full_build		- Full build process for the prod frontend (build, push, update manifest, deploy)."
	@echo ""
	@echo "Development and Production Deployment Targets:"
	@echo "  dev_deploy_all			- Deploy all development services (API and frontend)."
	@echo "  dev_api_integration_test		- Run integration tests for the dev API service."
	@echo ""
	@echo "Kind Cluster Targets:"
	@echo "  kind-init				- Initialize a Kind (Kubernetes in Docker) cluster."
	@echo "  kind-delete				- Delete the Kind cluster."
