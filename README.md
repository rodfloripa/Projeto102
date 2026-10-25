# MLOps Churn Pipeline - Olist

Pipeline de Machine Learning de ponta a ponta focado em robustez de engenharia, validação estrita de dados e monitoramento de *Data Drift*.

## Arquitetura
`Olist CSVs` -> `Validação (Pandera)` -> `Treinamento (Scikit-Learn + MLflow)` -> `API de Inferência (FastAPI + Docker)` -> `Monitoramento de Drift (Evidently)`

## Como Executar

1. Coloque os arquivos do Olist (`olist_orders_dataset.csv` e `olist_order_items_dataset.csv`) dentro da pasta `data/`.
2. Instale as dependências e rode o treino:
   ```bash
   pip install -r requirements.txt
   python src/train.py
   ```
3. Simule um lote atual e verifique o Data Drift:
   ```bash
   python -c "import pandas as pd; df = pd.read_csv('data/reference_data.csv'); df['recency_days'] *= 1.5; df.to_csv('data/current_batch.csv', index=False)"
   python src/drift.py
   ```
4. Suba a API e o MLflow via Docker Compose:
   ```bash
   docker-compose up --build
   ```
