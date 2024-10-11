docker build --file dockerfiles/dockerfile.classifier_server -t aslon1213/classifier_server .
docker container stop classifier_server && docker container rm classifier_server
docker run -d -p 50052:50050 --net consultant_ai --name classifier_server aslon1213/classifier_server:latest













