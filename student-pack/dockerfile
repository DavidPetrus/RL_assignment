FROM shahil/nvidia-docker-rl:assignment

WORKDIR /usr/src/

COPY . .

RUN Xvfb :1 -screen 0 800x600x16  &

CMD python3 evaluation.py