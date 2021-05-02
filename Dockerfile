FROM python:3.8

WORKDIR test

COPY cnc_ai cnc_ai
COPY README.md .
COPY setup.py .

RUN ["pip", "install", "."]

CMD cd && python -m cnc_ai.background_model -h
