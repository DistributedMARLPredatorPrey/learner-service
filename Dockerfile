FROM python:3.11

WORKDIR /usr/app

COPY . .
RUN pip install .

CMD [ "python3", "src/main/view/main.py"]
