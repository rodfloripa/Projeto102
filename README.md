# MLOps Churn Pipeline — Olist

<p align="justify">Pipeline de aprendizado de máquina de ponta a ponta, com foco em robustez de engenharia, validação rigorosa de dados, monitoramento de desvio de dados (<em>Data Drift</em>) e conteinerização.</p>

## 1. Arquitetura

<p align="justify"><code>Olist CSVs</code> → <code>Validação (Pandera)</code> → <code>Treinamento (Scikit-Learn + MLflow)</code> → <code>API de Inferência (FastAPI + Docker)</code> → <code>Monitoramento de Drift</code></p>

## 2. Estrutura do Projeto

```text
churn-mlops-olist/
├── data/                  # Diretório para conjuntos de dados brutos e de referência (ignorado pelo Git)
│   └── .gitkeep
├── model/                 # Artefatos de modelos treinados salvos localmente
│   └── model.pkl
├── src/                   # Código-fonte principal do pipeline
│   ├── api.py             # Aplicação FastAPI para disponibilizar o modelo
│   ├── drift.py           # Detecção de desvio de dados pelo teste de Kolmogorov-Smirnov
│   ├── train.py           # Engenharia de atributos, treinamento e registro no MLflow
│   └── validate.py        # Esquemas de validação de dados com Pandera
├── .gitignore             # Arquivos e pastas ignorados pelo controle de versão
├── Dockerfile             # Configuração da imagem Docker da API de inferência
├── docker-compose.yml     # Orquestração dos contêineres da API e do MLflow
├── requirements.txt       # Dependências Python do projeto
└── README.md              # Documentação oficial do projeto
```

## 3. Responsabilidade de Cada Arquivo

### 3.1. `src/validate.py` — Validação dos Dados

<p align="justify">Utiliza a biblioteca <code>Pandera</code> para aplicar contratos rigorosos de validação aos conjuntos de dados brutos do Olist antes do processamento. O objetivo é garantir a integridade dos tipos, a ausência de valores nulos em colunas críticas e o respeito aos limites numéricos definidos.</p>

### 3.2. `src/train.py` — Treinamento do Modelo

<p align="justify">É o script central de aprendizado de máquina. Realiza a ingestão dos dados validados, a engenharia de atributos RFM — recência, frequência e valor monetário —, o treinamento do modelo <code>RandomForestClassifier</code>, a avaliação de métricas e o versionamento automático de parâmetros e artefatos por meio do <code>MLflow</code>.</p>

### 3.3. `src/drift.py` — Monitoramento de Desvio de Dados

<p align="justify">Executa o teste de Kolmogorov-Smirnov, por meio da função <code>ks_2samp</code>, para comparar a distribuição dos atributos do conjunto de referência (<code>reference_data.csv</code>) com a distribuição do lote atual (<code>current_batch.csv</code>). O processo gera alertas visuais e um relatório HTML chamado <code>drift_report.html</code>.</p>

### 3.4. `src/api.py` — API de Inferência

<p align="justify">Implementa uma API REST com <code>FastAPI</code>, que carrega o modelo treinado e disponibiliza previsões de cancelamento de clientes (<em>churn</em>) em tempo real. A tipagem e a validação das entradas são realizadas com <code>Pydantic</code>.</p>

### 3.5. `Dockerfile` — Imagem da API

<p align="justify">Define o ambiente isolado em contêiner para empacotar a API e suas dependências, buscando uma execução leve e reproduzível em diferentes ambientes compatíveis com Docker.</p>

### 3.6. `docker-compose.yml` — Orquestração dos Serviços

<p align="justify">Coordena a inicialização conjunta dos serviços em contêineres, incluindo a API de inferência e o servidor de rastreamento de experimentos do <code>MLflow</code>.</p>

### 3.7. Arquivos de Apoio

<p align="justify"><code>data/</code> armazena os conjuntos de dados brutos e de referência utilizados pelo pipeline; <code>model/</code> guarda localmente os artefatos do modelo treinado; <code>.gitignore</code> define o que não deve ser acompanhado pelo controle de versão; e <code>requirements.txt</code> lista as dependências Python necessárias para executar o projeto. O arquivo <code>.gitkeep</code> permite manter o diretório <code>data/</code> no repositório mesmo quando seus arquivos de dados não são versionados.</p>

## 4. Como Executar

### 4.1. Preparar os Dados

<p align="justify">Coloque os arquivos CSV do conjunto de dados Olist — <code>olist_orders_dataset.csv</code> e <code>olist_order_items_dataset.csv</code> — dentro do diretório <code>data/</code>.</p>

### 4.2. Instalar as Dependências e Treinar

<p align="justify">Instale as bibliotecas listadas em <code>requirements.txt</code> e execute o script de treinamento a partir da raiz do projeto:</p>

```bash
pip install -r requirements.txt
python src/train.py
```

### 4.3. Simular um Lote Atual e Verificar o Desvio de Dados

<p align="justify">O comando a seguir cria um lote de teste a partir dos dados de referência, multiplica por 1,5 a coluna <code>recency_days</code> e salva o resultado como <code>current_batch.csv</code>. Em seguida, executa o monitoramento de desvio de dados:</p>

```bash
python -c "import pandas as pd; df = pd.read_csv('data/reference_data.csv'); df['recency_days'] *= 1.5; df.to_csv('data/current_batch.csv', index=False)"
python src/drift.py
```

### 4.4. Iniciar a API e os Serviços

<p align="justify">Na raiz do projeto, execute o Docker Compose para construir a imagem e iniciar os serviços definidos na configuração:</p>

```bash
docker-compose up --build

Chamando a API
<p align="center">
  <img src="https://github.com/rodfloripa/Projeto102/blob/main/fig0.png">
</p>```

## 5. Fluxo de Execução

<p align="justify">O fluxo começa com os arquivos CSV do Olist. Os dados passam pela validação com <code>Pandera</code> antes de serem utilizados na engenharia de atributos e no treinamento do modelo com <code>Scikit-Learn</code>. Os parâmetros e artefatos dos experimentos são registrados no <code>MLflow</code>. O modelo treinado é disponibilizado por uma API criada com <code>FastAPI</code> e empacotada em um contêiner Docker. Em paralelo, o módulo de monitoramento compara os atributos de referência com os de um lote atual usando o teste de Kolmogorov-Smirnov e produz um relatório sobre possíveis mudanças nas distribuições.</p>

## 6. Observações de Execução

<p align="justify">A execução dos comandos depende da presença dos arquivos de dados esperados e dos artefatos produzidos nas etapas anteriores. Em particular, a simulação de desvio pressupõe que <code>data/reference_data.csv</code> já exista e contenha a coluna <code>recency_days</code>. O código de treinamento, a API e o monitoramento devem utilizar nomes de arquivos e atributos consistentes entre si.</p>

<p align="justify">A árvore de diretórios e as descrições acima documentam a estrutura prevista para o projeto fornecido. Para confirmar detalhes de implementação — como os nomes exatos dos campos de entrada da API, os critérios numéricos de validação, as métricas do modelo e os limiares usados para sinalizar desvio — é necessário consultar os respectivos arquivos Python.</p>
