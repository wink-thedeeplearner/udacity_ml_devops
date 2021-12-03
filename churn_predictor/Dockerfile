FROM python:3.9.6-slim-buster

COPY ./ /churn_predictor
WORKDIR /churn_predictor

RUN pip install -r requirements.txt

VOLUME /churn_predictor/logs
VOLUME /churn_predictor/plot_figures
VOLUME /churn_predictor/models

RUN mkdir -p /churn_predictor/logs && \
    mkdir -p /churn_predictor/models && \
    mkdir -p /churn_predictor/plot_figures/eda && \
    mkdir -p /churn_predictor/plot_figures/results

ENTRYPOINT ["python"]
CMD ["churn_predictor_test.py"]
