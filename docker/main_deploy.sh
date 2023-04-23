bash install_docker.sh

sudo docker build --no-cache -t layoutparser-api:0.1 .

sudo docker run -it -d --name layoutparser-api-serv -p 8888:8888 -p 8889:8889 layoutparser-api:0.1
