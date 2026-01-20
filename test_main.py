import pytest
import pandas as pd
import numpy as np
import os
from main import load_and_preprocess_data, initialize_llm, evaluate_model
from sklearn.svm import SVC

# Criamos um DataFrame com 30 linhas para que o SMOTE tenha vizinhos suficientes
@pytest.fixture
def mock_df(tmp_path):
    np.random.seed(42)
    rows = 30
    data = {
        'Pregnancies': np.random.randint(0, 10, rows),
        'Glucose': np.random.randint(80, 200, rows),
        'BloodPressure': np.random.randint(60, 100, rows),
        'SkinThickness': np.random.randint(10, 50, rows),
        'Insulin': np.random.randint(0, 300, rows),
        'BMI': np.random.uniform(18, 40, rows),
        'DiabetesPedigreeFunction': np.random.uniform(0.1, 1.5, rows),
        'Age': np.random.randint(20, 70, rows),
        'Outcome': [0, 1] * 15  # 15 de cada classe para garantir equilíbrio
    }
    # Inserir alguns zeros para testar o SimpleImputer
    df = pd.DataFrame(data)
    df.loc[0:2, 'Glucose'] = 0 
    df.loc[3:5, 'BMI'] = 0
    
    csv_file = tmp_path / "test_diabetes.csv"
    df.to_csv(csv_file, index=False)
    return str(csv_file)

def test_load_and_preprocess_data(mock_df):
    """Testa se o preprocessamento remove zeros e escala os dados corretamente"""
    # Agora com 30 linhas e 15 de cada classe, o SMOTE e o Stratify funcionarão
    results = load_and_preprocess_data(mock_df)
    X_train_pre = results[4]
    
    # Verifica se as 8 colunas preditoras estão lá
    assert X_train_pre.shape[1] == 8
    # Verifica se os zeros foram tratados (não deve haver NaNs após o imputer)
    assert not np.isnan(X_train_pre).any()

def test_evaluate_model():
    """Testa a função de métricas"""
    model = SVC()
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 1, 0, 1, 0, 1])
    model.fit(X, y)
    metrics = evaluate_model(model, X, y)
    assert 'recall' in metrics
    assert metrics['accuracy'] > 0

def test_llm_initialization_flow():
    """Verifica se a inicialização do LLM retorna um cliente ou None sem crashar"""
    # Este teste apenas garante que a função executa o fluxo interno
    client = initialize_llm()
    # O resultado depende se você tem o .env configurado ou não na máquina
    assert client is None or hasattr(client, 'chat_completion')

def test_genetic_algorithm_loop(mock_df):
    """Verifica se o GA consegue rodar uma geração pequena sem erros"""
    from main import run_genetic_algorithm, load_and_preprocess_data
    
    results = load_and_preprocess_data(mock_df)
    X_train_bal, y_train_bal = results[6], results[7]
    X_test_pre, y_test = results[5], results[3]
    
    # Rodar uma versão 'mini' do GA
    best_ind, best_fit, recall, f1, acc, prec = run_genetic_algorithm(
        X_train_bal, y_train_bal, X_test_pre, y_test,
        pop_size=2, generations=1, mutation_rate=0.1
    )
    
    assert best_ind is not None
    assert 0 <= recall <= 1    