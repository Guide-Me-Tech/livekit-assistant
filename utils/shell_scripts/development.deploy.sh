docker build --file dockerfiles/dockerfile.development -t aslon1213/classifier_development .
docker container stop classifier && docker container rm classifier
docker run -d -p 50051:50051 -p 50050:50050 --mount type=bind,source=/home/sbadmin/aslon/consultant_ai_sentence_classifier/,target=/usr/src/app/ --net consultant_ai --name classifier aslon1213/classifier_development