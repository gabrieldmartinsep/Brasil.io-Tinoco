import requests
import os
import json
import pandas as pd
from datetime import datetime
import time # Importar time para as pausas

# --- Configurações Iniciais ---
BASE_DIR = 'dataset'
RAW_PATH = os.path.join(BASE_DIR, 'raw')
BRONZE_PATH = os.path.join(BASE_DIR, 'bronze')

# URL base
API_URL = 'https://brasil.io/api/v1/dataset/gastos-diretos/gastos/data/'
API_TOKEN = 'fc7f6a051a8a5f9a6c9ad750aa0752b2c85ab531'
HEADERS = {'Authorization': f'Token {API_TOKEN}'}
RAW_FILENAME = 'govbr_all_pages.json'
MAX_PAGES_TO_FETCH = 1301 # Limite de páginas
PAUSE_SECONDS = 1.0 # Pausa normal entre requisições
PAUSE_ON_ERROR_429 = 60 # Pausa longa ao receber erro 429 (Limite de Requisição) (60 segundos)

def ensure_dirs():
    #Garante que as pastas raw, bronze, silver e gold existam.
    print("LOG: Verificando e criando a estrutura de pastas...")
    for folder in ['raw', 'bronze', 'silver', 'gold']:
        path = os.path.join(BASE_DIR, folder)
        os.makedirs(path, exist_ok=True)
    print("LOG: Estrutura de pastas pronta.")

def fetch_and_store_data():
    #Faz a chamada para a API, iterando pelas páginas, com pausa, e
    #continua de onde parou se o arquivo JSON existir.
    #Tenta novamente (auto-retry) em caso de erro 429.
    
    all_results = []
    filepath = os.path.join(RAW_PATH, RAW_FILENAME)
    start_page = 1
    
    # Logica de continuacao (Checkpoint)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                all_results.extend(existing_data)
                
            # A API retorna 100 resultados por página (assumindo 100).
            # Se a API retornar menos (ex: 50), este cálculo precisa ser ajustado.
            # Assumindo 100, baseado nas execuções anteriores (57000 / 57 = 1000, 30000 / 30 = 1000)
            # A API do Brasil.IO retorna 1000 por padrão, não 100.
            # 57000 registros / 57 páginas = 1000 por página.
            
            start_page = (len(all_results) // 1000) + 1 
            print(f"\n--- LOG: CHECKPOINT ENCONTRADO ---")
            print(f"LOG: {len(all_results)} registros já baixados no RAW.")
            print(f"LOG: Iniciando download da Página {start_page}.")
            print("----------------------------------\n")
            
        except Exception as e:
            print(f"LOG: Arquivo RAW existente corrompido ou vazio ({e}). Reiniciando download.")
            start_page = 1
    else:
        print("LOG: Nenhum checkpoint encontrado. Iniciando download da Página 1.")

    
    # O loop 'while' serve para controlar manualmente o número da página em caso de retry
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
                break # Encerra o loop 'while'
                
            all_results.extend(page_results)
            print(f"LOG: OK (Pág {page_num}). Total de registros na memória: {len(all_results)}")
            
            # Salva um arquivo de checkpoint a cada 50 páginas (persistência)
            if page_num % 50 == 0: 
                 print(f"LOG: CHECKPOINT: Salvando {len(all_results)} registros em JSON.")
                 with open(filepath, 'w', encoding='utf-8') as f:
                     json.dump(all_results, f, ensure_ascii=False, indent=4)
            
            # Pausa para evitar 429
            time.sleep(PAUSE_SECONDS) 
            
            # Incrementa a página SOMENTE se houver sucesso
            page_num += 1 
        
        elif response.status_code == 429:
            print(f"\nLOG: ERRO 429 (Pág {page_num}): Limite de Requisições.")
            print(f"LOG: Salvando checkpoint de {len(all_results)} registros...")
            
            # Salva o progresso
            with open(filepath, 'w', encoding='utf-8') as f:
                 json.dump(all_results, f, ensure_ascii=False, indent=4)
                 
            print(f"LOG: Checkpoint salvo. Pausando por {PAUSE_ON_ERROR_429} segundos...")
            # Pausa longa antes de tentar a MESMA página novamente
            time.sleep(PAUSE_ON_ERROR_429)
        
        else:
            # Erros 404, 500, etc.
            print(f"\nLOG: ERRO CRÍTICO (Pág {page_num}): Status Code {response.status_code}.")
            print(f"LOG: Abortando. Salvando checkpoint final...")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                 json.dump(all_results, f, ensure_ascii=False, indent=4)
            
            break # Encerra o loop 'while'

    if not all_results:
        print("LOG: Nenhum dado foi baixado com sucesso após tentativas.")
        return None

    # Salva todos os resultados em um único arquivo JSON
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
    #Lê o JSON, transforma em Parquet e particiona por ano e mês.
    #Usa uma lista de colunas prováveis para encontrar a data correta.

    if not json_filepath or not os.path.exists(json_filepath):
        print("LOG: Caminho do arquivo JSON não encontrado ou vazio. Abortando transformação.")
        return

    print(f"LOG: Iniciando transformação e particionamento para: {json_filepath}")
    
    # 1. Carregar os Dados JSON
    try:
        df = pd.read_json(json_filepath) 
    except Exception as e:
        print(f"LOG: Erro ao ler o arquivo JSON: {e}")
        return

    if df.empty:
        print("LOG: DataFrame lido está vazio. Abortando particionamento.")
        return

    #Remover colunas 'ano' e 'mes' existentes para evitar conflito
    cols_to_drop = [c for c in ['ano', 'mes'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"LOG: Colunas existentes removidas para evitar conflito: {cols_to_drop}")

    # 2. Preparar para Particionamento 
    date_columns = ['data_pagamento', 'data_pagamento_original', 'data_transacao', 'data_empenho', 'data_lancamento', 'date'] 
    data_col_name = None

    for col in date_columns:
        if col in df.columns:
            data_col_name = col
            break

    if data_col_name:
        print(f"LOG: Coluna de data encontrada: '{data_col_name}'.")
        
        df['date_obj'] = pd.to_datetime(df[data_col_name], errors='coerce') 
        
        # Filtra registros onde a data foi inválida
        df = df.dropna(subset=['date_obj']) 
        
        if df.empty:
            print("LOG: AVISO: DataFrame ficou vazio após a limpeza de datas inválidas. Abortando.")
            return

        df['ano'] = df['date_obj'].dt.strftime('%Y')
        df['mes'] = df['date_obj'].dt.strftime('%m')
    else:
        print("LOG: ERRO DE COLUNA: Nenhuma coluna de data padrão foi encontrada. Particionamento abortado.")
        print(f"LOG: Colunas disponíveis após limpeza: {df.columns.tolist()}")
        return
    
    # 3. Salvar como Parquet Particionado
    try:
        print(f"LOG: Iniciando escrita na Camada Bronze...")
        df.to_parquet(
            path=BRONZE_PATH,
            engine='pyarrow',
            partition_cols=['ano', 'mes'],
            index=False,
        )
        print(f"Dados transformados e particionados em Parquet na pasta: {BRONZE_PATH}")
    except Exception as e:
        print(f"ERRO GRAVE DE ESCRITA: Não foi possível salvar o Parquet. Detalhes: {e}")

def main():
    #Função principal para conduzir as etapas.
    ensure_dirs()
    
    # Etapa 1: Chamada à API e armazenamento em RAW
    raw_file = fetch_and_store_data()
    
    # Etapa 2: Processamento e armazenamento em BRONZE
    if raw_file:
        transform_to_parquet_and_partition(raw_file)
        
    print("\nProjeto concluído com sucesso!")
    print(f"Verifique os dados brutos em: {RAW_PATH}/{RAW_FILENAME}")
    print(f"Verifique os dados particionados em: {BRONZE_PATH}")

if __name__ == "__main__":
    main()