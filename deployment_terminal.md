```
[shiv@pc fraud-detection master]$make kind-init 
kind create cluster --config kind.config
Creating cluster "kind" ...
 ✓ Ensuring node image (kindest/node:v1.27.3) 🖼
 ✓ Preparing nodes 📦  
 ✓ Writing configuration 📜 
 ✓ Starting control-plane 🕹 
 ✓ Installing CNI 🔌 
 ✓ Installing StorageClass 💾 
Set kubectl context to "kind-kind"
You can now use your cluster with:

kubectl cluster-info --context kind-kind

Have a nice day! 👋
sleep 5
kubectl apply -f https://kind.sigs.k8s.io/examples/ingress/deploy-ingress-nginx.yaml
namespace/ingress-nginx created
serviceaccount/ingress-nginx created
serviceaccount/ingress-nginx-admission created
role.rbac.authorization.k8s.io/ingress-nginx created
role.rbac.authorization.k8s.io/ingress-nginx-admission created
clusterrole.rbac.authorization.k8s.io/ingress-nginx created
clusterrole.rbac.authorization.k8s.io/ingress-nginx-admission created
rolebinding.rbac.authorization.k8s.io/ingress-nginx created
rolebinding.rbac.authorization.k8s.io/ingress-nginx-admission created
clusterrolebinding.rbac.authorization.k8s.io/ingress-nginx created
clusterrolebinding.rbac.authorization.k8s.io/ingress-nginx-admission created
configmap/ingress-nginx-controller created
service/ingress-nginx-controller created
service/ingress-nginx-controller-admission created
deployment.apps/ingress-nginx-controller created
job.batch/ingress-nginx-admission-create created
job.batch/ingress-nginx-admission-patch created
ingressclass.networking.k8s.io/nginx created
validatingwebhookconfiguration.admissionregistration.k8s.io/ingress-nginx-admission created
sleep 30
kubectl wait --namespace ingress-nginx \
--for=condition=ready pod \
--selector=app.kubernetes.io/component=controller \
--timeout=90s
error: timed out waiting for the condition on pods/ingress-nginx-controller-75b98596c9-29mvl
make: *** [Makefile:234: kind-init] Error 1
[shiv@pc fraud-detection master]$kubectl get po -n ingress-nginx 
NAME                                        READY   STATUS      RESTARTS   AGE
ingress-nginx-admission-create-vtfrc        0/1     Completed   0          2m23s
ingress-nginx-admission-patch-dh7fp         0/1     Completed   1          2m23s
ingress-nginx-controller-75b98596c9-29mvl   1/1     Running     0          2m23s
[shiv@pc fraud-detection master]$
[shiv@pc fraud-detection master]$make dev_deploy_all
make dev_api_deploy 
make[1]: Entering directory '/run/media/shiv/e202b7b3-865c-4d22-9196-f1c9deb5d5f2/code/fraud-detection'
Applying Kubernetes manifests...
manifest/api/overlays/dev
kubectl apply -k manifest/api/overlays/dev 
configmap/dev-fraud-detection-api-config-5m9tgg6dtg created
service/dev-fraud-detection-api created
deployment.apps/dev-fraud-detection-api created
make[1]: Leaving directory '/run/media/shiv/e202b7b3-865c-4d22-9196-f1c9deb5d5f2/code/fraud-detection'
make dev_frontend_deploy
make[1]: Entering directory '/run/media/shiv/e202b7b3-865c-4d22-9196-f1c9deb5d5f2/code/fraud-detection'
Applying Kubernetes manifests...
manifest/frontend/overlays/dev
kubectl apply -k manifest/frontend/overlays/dev 
configmap/dev-fraud-detection-frontend-config-g86dk279kb created
service/dev-fraud-detection-frontend created
deployment.apps/dev-fraud-detection-frontend created
ingress.networking.k8s.io/dev-fraud-detection-frontend-ingress created
make[1]: Leaving directory '/run/media/shiv/e202b7b3-865c-4d22-9196-f1c9deb5d5f2/code/fraud-detection'
[shiv@pc fraud-detection master]$kubectl get po
NAME                                            READY   STATUS              RESTARTS   AGE
dev-fraud-detection-api-689f98fcfc-wp5g6        0/1     ContainerCreating   0          13s
dev-fraud-detection-frontend-54c584d7b9-2dz8z   0/1     ContainerCreating   0          13s
[shiv@pc fraud-detection master]$kubectl get po -w
NAME                                            READY   STATUS              RESTARTS   AGE
dev-fraud-detection-api-689f98fcfc-wp5g6        0/1     ContainerCreating   0          55s
dev-fraud-detection-frontend-54c584d7b9-2dz8z   0/1     ContainerCreating   0          55s
dev-fraud-detection-api-689f98fcfc-wp5g6        1/1     Running             0          13m
dev-fraud-detection-frontend-54c584d7b9-2dz8z   1/1     Running             0          16m
^C[shiv@pc fraud-detection master]$make dev_api_integation_test
**Note******
Make sure virtual env is active and pytest is installed

=========================================================================================================== test session starts ============================================================================================================
platform linux -- Python 3.12.7, pytest-8.3.3, pluggy-1.5.0
rootdir: /run/media/shiv/e202b7b3-865c-4d22-9196-f1c9deb5d5f2/code/fraud-detection/services/fraud_detection/integration
plugins: rerunfailures-14.0, typeguard-4.3.0, time-machine-2.16.0, cov-5.0.0, repeat-0.9.3
collected 1 item                                                                                                                                                                                                                           

test_integration.py .                                                                                                                                                                                                                [100%]

============================================================================================================ 1 passed in 0.07s =============================================================================================================
[shiv@pc fraud-detection master]$
```