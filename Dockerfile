FROM python:3.8

RUN ["pip", "install", "git+https://github.com/gaebor/CnC_AI.git"]

CMD ["python", "-m", "cnc_ai.background_model", "-h"]
