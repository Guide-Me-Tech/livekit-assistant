docker build --file dockerfiles/dockerfile.classifier_server.development -t aslon1213/classifier_server_dev .
docker container stop classifier_server_dev && docker container rm classifier_server_dev
docker run -d --net consultant_ai --name classifier_server_dev aslon1213/classifier_server_dev:latest













