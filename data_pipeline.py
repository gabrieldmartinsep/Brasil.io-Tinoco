import requests
import os
import json
import pandas as pd
from datetime import datetime
import time 
import shutil # Novo import para remover pastas

# --- Configurações de Caminho e API ---
BASE_DIR = 'dataset'
RAW_PATH = os.path.join(BASE_DIR, 'raw')
BRONZE_PATH = os.path.join(BASE_DIR, 'bronze')
SILVER_PATH = os.path.join(BASE_DIR, 'silver') # Novo Caminho
GOLD_PATH = os.path.join(BASE_DIR, 'gold') # Novo Caminho

# Configurações de API (Mantidas do seu código original)
API_URL = 'https://brasil.io/api/v1/dataset/gastos-diretos/gastos/data/'
API_TOKEN = 'fc7f6a051a8a5f9a6c9ad750aa0752b2c85ab531'
HEADERS = {'Authorization': f'Token {API_TOKEN}'}
RAW_FILENAME = 'govbr_all_pages.json'
MAX_PAGES_TO_FETCH = 1301 
PAUSE_SECONDS = 1.0 
PAUSE_ON_ERROR_429 = 60 
SILVER_FILENAME = 'gastos_limpos.parquet'
GOLD_FILENAME = 'gastos_agregados_mensais.parquet'

def ensure_dirs():
    # Garante que as pastas raw, bronze, silver e gold existam.
    print("LOG: Verificando e criando a estrutura de pastas...")
    for folder in ['raw', 'bronze', 'silver', 'gold']:
        path = os.path.join(BASE_DIR, folder)
        os.makedirs(path, exist_ok=True)
    print("LOG: Estrutura de pastas pronta.")

def fetch_and_store_data():
    # Faz a chamada para a API, iterando pelas páginas, com checkpointing e auto-retry em caso de erro 429. Armazena na camada RAW.
    
    all_results = []
    filepath = os.path.join(RAW_PATH, RAW_FILENAME)
    start_page = 1
    
    # Lógica de continuacao (Checkpoint)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                all_results.extend(existing_data)
                
            # Assumindo 1000 registros por página
            start_page = (len(all_results) // 1000) + 1 
            
            # Se o download estiver completo, apenas retorna o caminho
            if start_page > MAX_PAGES_TO_FETCH:
                 print("\n--- LOG: DOWNLOAD RAW COMPLETO ---")
                 print(f"LOG: Todas as {MAX_PAGES_TO_FETCH} páginas já foram baixadas. Pulando download.")
                 print("----------------------------------\n")
                 return filepath
                 
            print(f"\n--- LOG: CHECKPOINT ENCONTRADO ---")
            print(f"LOG: {len(all_results)} registros já baixados no RAW.")
            print(f"LOG: Iniciando download da Página {start_page}.")
            print("----------------------------------\n")
            
        except Exception as e:
            print(f"LOG: Arquivo RAW existente corrompido ou vazio ({e}). Reiniciando download.")
            start_page = 1
    else:
        print("LOG: Nenhum checkpoint encontrado. Iniciando download da Página 1.")

    
    page_num = start_page
    
    while page_num <= MAX_PAGES_TO_FETCH:
        current_url = f"{API_URL}?page={page_num}"
        print(f"LOG: Chamando Página {page_num} de {MAX_PAGES_TO_FETCH}...")
        
        response = requests.get(current_url, headers=HEADERS)
        
        if response.status_code == 200:
            data = response.json()
            page_results = data.get('results', [])
            
            if not page_results:
                print(f"LOG: Status 200, mas Página {page_num} não retornou resultados. Fim da paginação.")
                break 
                
            all_results.extend(page_results)
            print(f"LOG: OK (Pág {page_num}). Total de registros na memória: {len(all_results)}")
            
            # Salva um arquivo de checkpoint a cada 50 páginas (persistência)
            if page_num % 50 == 0: 
                print(f"LOG: CHECKPOINT: Salvando {len(all_results)} registros em JSON.")
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=4)
            
            time.sleep(PAUSE_SECONDS) 
            page_num += 1 
        
        elif response.status_code == 429:
            print(f"\nLOG: ERRO 429 (Pág {page_num}): Limite de Requisições.")
            print(f"LOG: Salvando checkpoint de {len(all_results)} registros...")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
                
            print(f"LOG: Checkpoint salvo. Pausando por {PAUSE_ON_ERROR_429} segundos...")
            time.sleep(PAUSE_ON_ERROR_429)
        
        else:
            print(f"\nLOG: ERRO CRÍTICO (Pág {page_num}): Status Code {response.status_code}.")
            print(f"LOG: Abortando. Salvando checkpoint final...")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
            
            break 

    if not all_results:
        print("LOG: Nenhum dado foi baixado com sucesso após tentativas.")
        return None

    try:
        print(f"\nLOG: Salvando arquivo RAW final (Total: {len(all_results)} registros).")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        
        print(f"Dados (total de {len(all_results)} registros) armazenados com sucesso em: {filepath}")
        return filepath
    except Exception as e:
        print(f"Erro ao salvar o JSON: {e}")
        return None


def transform_to_parquet_and_partition(json_filepath):
    # Lê o JSON da RAW, transforma em DataFrame, cria colunas de ano/mês e salva o Parquet particionado na camada BRONZE.

    if not json_filepath or not os.path.exists(json_filepath):
        print("LOG: Caminho do arquivo JSON não encontrado ou vazio. Abortando BRONZE.")
        return False

    print(f"\n## Etapa BRONZE: Transformação para Parquet e Particionamento ##")
    
    try:
        df = pd.read_json(json_filepath) 
    except Exception as e:
        print(f"LOG: Erro ao ler o arquivo JSON: {e}")
        return False

    if df.empty:
        print("LOG: DataFrame lido está vazio. Abortando particionamento BRONZE.")
        return False

    cols_to_drop = [c for c in ['ano', 'mes'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"LOG: Colunas de partição temporárias removidas: {cols_to_drop}")

    date_columns = ['data_pagamento', 'data_pagamento_original', 'data_transacao', 'data_empenho', 'data_lancamento', 'date'] 
    data_col_name = None

    for col in date_columns:
        if col in df.columns:
            data_col_name = col
            break

    if data_col_name:
        print(f"LOG: Coluna de data para particionamento: '{data_col_name}'.")
        df['date_obj'] = pd.to_datetime(df[data_col_name], errors='coerce') 
        df = df.dropna(subset=['date_obj']) 
        
        if df.empty:
            print("LOG: AVISO: DataFrame ficou vazio após a limpeza de datas inválidas. Abortando BRONZE.")
            return False

        df['ano'] = df['date_obj'].dt.strftime('%Y')
        df['mes'] = df['date_obj'].dt.strftime('%m')
    else:
        print("LOG: ERRO: Nenhuma coluna de data padrão foi encontrada. Particionamento abortado BRONZE.")
        return False
    
    try:
        print(f"LOG: Iniciando escrita na Camada Bronze (Total: {len(df)} registros)...")
        # NOTA: O Parquet particionado não deve ter o parâmetro basename
        df.to_parquet(
            path=BRONZE_PATH,
            engine='pyarrow',
            partition_cols=['ano', 'mes'],
            index=False,
        )
        print(f"Dados transformados e particionados em Parquet na pasta: {BRONZE_PATH}")
    except Exception as e:
        print(f"ERRO GRAVE DE ESCRITA: Não foi possível salvar o Parquet BRONZE. Detalhes: {e}")
        return False # Retorna falha
    return True


def process_bronze_to_silver():
    # Lê os dados do BRONZE, aplica Data Wrangling (limpeza e tipagem), executa um teste de qualidade simples e salva na camada SILVER.
    print("\n## Etapa SILVER: Refinamento e Qualidade dos Dados (Data Wrangling) ##")
    
    # 1. Carregar os Dados Particionados do Bronze
    try:
        # Lê o dataset inteiro particionado do BRONZE
        df = pd.read_parquet(BRONZE_PATH)
        print(f"LOG: Dados carregados do BRONZE. Total de registros: {len(df)}")
    except Exception as e:
        print(f"LOG: ERRO: Não foi possível ler os dados da pasta BRONZE ({e}). Abortando SILVER.")
        return False
    
    # 2. Renomear Colunas (Aliasing) para Consistência do Pipeline
    # Mapeamento do nome da API para o nome interno do pipeline
    rename_mapping = {
        'nome_orgao': 'orgao_nome',              # API 'nome_orgao' -> ETL 'orgao_nome'
        'valor': 'valor_pagamento',              # API 'valor' -> ETL 'valor_pagamento' (Fato)
        'nome_acao': 'descricao_acao'            # API 'nome_acao' -> ETL 'descricao_acao'
    }
    
    df.rename(columns=rename_mapping, inplace=True)
    
    print("LOG: Colunas renomeadas para: 'orgao_nome', 'valor_pagamento', 'descricao_acao'.")
        
    # Colunas que deveriam estar presentes no SILVER após o renomeamento
    expected_dim_fact_cols = ['orgao_nome', 'descricao_acao', 'valor_pagamento'] 
    
    # 3. Verifica se colunas críticas estão faltando (pós-renomeamento)
    missing_cols_check = [col for col in expected_dim_fact_cols if col not in df.columns]

    if missing_cols_check:
        print("\nLOG: ===================================================")
        print("LOG: AVISO CRÍTICO DE SCHEMA: Algumas colunas essenciais ainda estão faltando!")
        print(f"LOG: Colunas essenciais (pós-renomeamento) faltando: {missing_cols_check}")
        print("LOG: O pipeline continuará, mas o resultado final pode não ser o esperado.")
        print("LOG: ===================================================\n")
    
    # --- 4. Data Wrangling (Limpeza e Padronização) ---
    
    # Padronização de Colunas de Texto (Usando nomes internos do pipeline)
    text_cols_to_standardize = ['orgao_nome', 'orgao_sigla', 'descricao_acao', 'municipio_nome']
    for col in text_cols_to_standardize:
        if col in df.columns:
            # Garante que seja string, remove espaços e converte para caixa alta
            df[col] = df[col].astype(str).str.strip().str.upper()
            
    # Tratamento de Valores Nulos (Imputação simples para texto)
    if 'orgao_nome' in df.columns:
        df['orgao_nome'] = df['orgao_nome'].replace('NAN', 'ORGAO NAO IDENTIFICADO')
    
    if 'descricao_acao' in df.columns:
        df['descricao_acao'] = df['descricao_acao'].replace('NAN', 'ACAO NAO DETALHADA')
    
    # Tipagem de Dados (Conversão Correta de Tipos)
    
    # a) Valor Monetário
    monetary_col = 'valor_pagamento'
    if monetary_col in df.columns:
        # Força para tipo numérico (float)
        df[monetary_col] = pd.to_numeric(df[monetary_col], errors='coerce')
        # Remove registros onde a conversão falhou (valores monetários críticos)
        df = df.dropna(subset=[monetary_col])
        df[monetary_col] = df[monetary_col].round(2)
        print(f"LOG: Coluna '{monetary_col}' convertida para float e arredondada. {len(df)} registros restantes.")
    
    # b) Códigos (Transformar para string para manter zeros iniciais, se aplicável)
    code_cols = ['municipio_codigo', 'unidade_gestora']
    for col in code_cols:
        if col in df.columns:
            # Converte para string para evitar que o Pandas interprete como int e perca o formato
            df[col] = df[col].astype(str).str.strip()
            
    # --- 5. Testes de Qualidade Simples (EDA - Valor do Negócio) ---
    print("\nLOG: == Análise Exploratória Simples (Valor do Negócio) ==")
    
    total_gasto = 0
    if monetary_col in df.columns:
        neg_count = (df[monetary_col] < 0).sum()
        total_gasto = df[monetary_col].sum()
        print(f"  - {monetary_col}: Registros negativos: {neg_count}. Total Gasto: R$ {total_gasto:,.2f}.")
    
    if 'date_obj' in df.columns and not df['date_obj'].empty:
        min_date = df['date_obj'].min().strftime('%Y-%m-%d')
        max_date = df['date_obj'].max().strftime('%Y-%m-%d')
        print(f"  - Período de Análise: De {min_date} a {max_date}.")
    else:
        print("  - Período de Análise: Dados de data ausentes ou vazios.")
        
    print("LOG: ===================================================")
    
    if monetary_col not in df.columns or total_gasto <= 0:
        print("LOG: ERRO DE QUALIDADE: Coluna de FATO (valor_pagamento) ausente ou total gasto <= 0. Abortando SILVER.")
        return False
        
    # --- 6. Seleção Final de Colunas (Definindo o Schema SILVER) ---
    silver_schema = [
        'date_obj', # Data completa (para agregação)
        'ano', 'mes', # Colunas de partição
        'valor_pagamento', # Fato principal
        'orgao_nome', 'orgao_sigla', # Dimensão: Órgão (orgao_sigla é do API)
        'municipio_nome',
        'descricao_acao', # Dimensão: Detalhe da Despesa
        'unidade_gestora',
    ]
    final_cols = [col for col in silver_schema if col in df.columns]
    df_silver = df[final_cols]
    
    # --- 7. Persistir no Silver ---
    try:
        print(f"\nLOG: Iniciando escrita na Camada SILVER (Total: {len(df_silver)} registros)...")
        df_silver.to_parquet(
            path=SILVER_PATH,
            engine='pyarrow',
            partition_cols=['ano', 'mes'], # Mantém o particionamento
            index=False,
        )
        print(f"Dados de ALTA QUALIDADE armazenados em: {SILVER_PATH}")
    except Exception as e:
        print(f"ERRO GRAVE DE ESCRITA: Não foi possível salvar o Parquet SILVER. Detalhes: {e}")
        return False
    
    return True


def process_silver_to_gold():
    #Lê os dados limpos do SILVER, aplica agregação (Transformação em Lote) para criar um artefato de dados de alto valor (produtos de dados) esalva na camada GOLD (Serviço).
    print("\n## Etapa GOLD: Agregação e Criação de Artefatos de Dados (Serving Layer) ##")
    
    # 1. Carregar os Dados Limpos do Silver
    try:
        df_silver = pd.read_parquet(SILVER_PATH)
        print(f"LOG: Dados carregados do SILVER. Total de registros: {len(df_silver)}")
    except Exception as e:
        print(f"LOG: ERRO: Não foi possível ler os dados da pasta SILVER ({e}). Abortando GOLD.")
        return
        
    if df_silver.empty:
        print("LOG: DataFrame SILVER vazio. Abortando GOLD.")
        return
        
    # --- 2. Aplicação da Transformação (Agregação para BI/Analytics) ---
    print("LOG: Aplicando transformação: Agregando Gasto Total Mensal por Órgão e Município...")
    
    required_cols = ['date_obj', 'ano', 'orgao_nome', 'valor_pagamento']
    
    # Verificação de colunas faltantes para logging
    missing_cols = [col for col in required_cols if col not in df_silver.columns]
    
    if missing_cols:
        print(f"LOG: ERRO: Colunas necessárias para agregação GOLD estão faltando: {missing_cols}. Abortando.")
        return
        
    # Cria uma chave de agregação (Ano-Mês)
    df_silver['periodo_mensal'] = df_silver['date_obj'].dt.to_period('M').astype(str)
    
    # Agregação: Gasto Total, Contagem de Transações
    # Agrupando por Município, pois UF não está presente na fonte
    groupby_cols = ['ano', 'periodo_mensal', 'orgao_nome']
    if 'municipio_nome' in df_silver.columns:
        groupby_cols.append('municipio_nome')

    df_gold = df_silver.groupby(groupby_cols, observed=False).agg(
        total_gasto_mensal=('valor_pagamento', 'sum'),
        total_transacoes_mensal=('valor_pagamento', 'count'),
        primeira_data_reg=('date_obj', 'min'),
        ultima_data_reg=('date_obj', 'max')
    ).reset_index()
    
    # Limpeza e Formatação da Gold Table
    df_gold['total_gasto_mensal'] = df_gold['total_gasto_mensal'].round(2)
    
    # --- 3. Salvar na Camada GOLD ---
    try:
        print(f"LOG: Iniciando escrita na Camada GOLD (Total de Linhas Agregadas: {len(df_gold)})...")
        df_gold.to_parquet(
            path=GOLD_PATH,
            engine='pyarrow',
            partition_cols=['ano'], 
            index=False,
        )
        print(f"Artefato de dados (Agregado Mensal) pronto para BI/ML em: {GOLD_PATH}")
    except Exception as e:
        print(f"ERRO GRAVE DE ESCRITA: Não foi possível salvar o Parquet GOLD. Detalhes: {e}")
        
    print("LOG: Etapa GOLD concluída.")


def main():
    # Função principal para conduzir as etapas do pipeline.
    ensure_dirs()
    
    # Etapa 1: Chamada à API e armazenamento em RAW
    raw_file = fetch_and_store_data()
    
    # Flag para controlar se podemos avançar para BRONZE e SILVER
    bronze_success = False
    
    if raw_file:
        # AÇÃO DE LIMPEZA: Se o RAW estiver completo, limpamos o BRONZE e SILVER antes de começar
        # Isso garante que não teremos arquivos com esquemas misturados de execuções anteriores.
        if os.path.exists(BRONZE_PATH):
            shutil.rmtree(BRONZE_PATH)
            os.makedirs(BRONZE_PATH) # Recria a pasta vazia
            print(f"LOG: LIMPEZA: Pasta BRONZE ({BRONZE_PATH}) limpa para recriação.")
            
        if os.path.exists(SILVER_PATH):
            shutil.rmtree(SILVER_PATH)
            os.makedirs(SILVER_PATH) # Recria a pasta vazia
            print(f"LOG: LIMPEZA: Pasta SILVER ({SILVER_PATH}) limpa para recriação.")

        # Etapa 2: Processamento e armazenamento em BRONZE
        bronze_success = transform_to_parquet_and_partition(raw_file)
    
    silver_success = False
    if bronze_success:
        # Etapa 3: Refinamento e armazenamento em SILVER
        silver_success = process_bronze_to_silver()
    if silver_success:
        # Etapa 4: Agregação e armazenamento em GOLD
        process_silver_to_gold()
        
    print("\n#################################################")
    print("### PROJETO ETL COMPLETO (RAW -> BRONZE -> SILVER -> GOLD) ###")
    print("#################################################")
    print(f"Dados Brutos (RAW): {RAW_PATH}/{RAW_FILENAME}")
    print(f"Dados Particionados (BRONZE): {BRONZE_PATH}")
    print(f"Dados de Qualidade (SILVER): {SILVER_PATH}")
    print(f"Artefato de Dados (GOLD - Agregado Mensal): {GOLD_PATH}")

if __name__ == "__main__":
    main()