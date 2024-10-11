docker container stop classifier && docker container rm classifier
docker run -d -p 50051:50051 --mount type=bind,source=/home/sbadmin/aslon/consultant_ai_sentence_classifier/,target=/usr/src/app/ --net consultant_ai --name classifier aslon1213/classifier
