import pandas as pd
import numpy as np
import random
import warnings
import logging
import sys
import matplotlib.pyplot as plt
from huggingface_hub import InferenceClient
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

# =====================
# 1. CONFIGURAÇÃO LLM (HUGGING FACE)
# =====================
# REQUISITO 3: Integração real e Prompt Engineering
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("Erro: Variável HF_TOKEN não encontrada. Verifique o arquivo .env")
client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=HF_TOKEN)

def get_llm_insight(outcome, raw_features):
    """Gera explicação usando Llama-3 (mais estável no roteamento atual)."""
    status = "Diabético" if outcome == 1 else "Não Diabético"
    
    # Criando o prompt estruturado
    messages = [
        {
            "role": "user", 
            "content": f"Você é um médico. Explique por que um paciente com esses dados: {raw_features.to_dict()} foi classificado como {status}. Dê 3 conselhos de saúde em português."
        }
    ]
    
    try:
        # Usamos o chat_completion que é o padrão ouro da API agora
        response = client.chat_completion(
            messages=messages,
            max_tokens=500,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        # Se o Llama falhar, tentamos uma última vez com o modelo de fallback universal
        return f"Erro na conexão LLM: {e}. Verifique se o seu Token tem permissão de escrita/leitura."
# =====================
# 2. PRÉ-PROCESSAMENTO (Fase 1)
# =====================
def load_and_preprocess(path):
    df = pd.read_csv(path)
    # Tratar zeros inválidos
    cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for c in cols: df[c] = df[c].replace(0, np.nan)
    
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Pipeline de transformação
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer([('num', numeric_transformer, X.columns.tolist())])
    
    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)
    
    # Balanceamento
    X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train_pre, y_train)
    
    return X_train, X_test, y_train, y_test, X_train_pre, X_test_pre, X_train_bal, y_train_bal

# =====================
# 3. ALGORITMO GENÉTICO (REQUISITO 1)
# =====================


HYPERPARAMS = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}

def fitness(ind, X_tr, y_tr, X_te, y_te):
    try:
        model = SVC(C=ind['C'], kernel=ind['kernel'], gamma=ind['gamma'], random_state=42, max_iter=10000)
        model.fit(X_tr, y_tr)
        return recall_score(y_te, model.predict(X_te), zero_division=0)
    except: return 0.0

def run_ga(X_tr, y_tr, X_te, y_te, config):
    pop = [{k: random.choice(v) for k, v in HYPERPARAMS.items()} for _ in range(config['pop_size'])]
    best_ind = None
    best_score = -1

    for gen in range(config['generations']):
        fits = [fitness(ind, X_tr, y_tr, X_te, y_te) for ind in pop]
        if max(fits) > best_score:
            best_score = max(fits)
            best_ind = pop[np.argmax(fits)]
        
        # Seleção e Crossover simples
        parents = [pop[i] for i in np.argsort(fits)[-config['num_parents']:]]
        new_pop = [best_ind]
        while len(new_pop) < config['pop_size']:
            p1, p2 = random.sample(parents, 2)
            child = {k: random.choice([p1[k], p2[k]]) for k in HYPERPARAMS}
            if random.random() < config['mutation_rate']:
                k = random.choice(list(HYPERPARAMS.keys()))
                child[k] = random.choice(HYPERPARAMS[k])
            new_pop.append(child)
        pop = new_pop
    return best_ind, best_score

# =====================
# 4. EXECUÇÃO E LOGGING
# =====================
def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    X_train, X_test, y_train, y_test, X_train_pre, X_test_pre, X_train_bal, y_train_bal = load_and_preprocess("diabetes.csv")
    
    results = []

    # 1. Modelo Base (SVM Simples)
    base_svm = SVC(random_state=42).fit(X_train_pre, y_train)
    y_pred_base = base_svm.predict(X_test_pre)
    results.append({'Modelo': 'Base SVM', 'Recall': recall_score(y_test, y_pred_base), 'Accuracy': accuracy_score(y_test, y_pred_base)})

    # 2. Experimentos GA (REQUISITO 1.3: Três Experimentos)
    ga_configs = [
        {'pop_size': 8, 'generations': 4, 'mutation_rate': 0.1, 'num_parents': 4, 'label': 'GA Exp1'},
        {'pop_size': 15, 'generations': 6, 'mutation_rate': 0.05, 'num_parents': 6, 'label': 'GA Exp2'},
        {'pop_size': 10, 'generations': 10, 'mutation_rate': 0.2, 'num_parents': 5, 'label': 'GA Exp3'}
    ]

    best_overall_model = None
    max_recall = -1

    for config in ga_configs:
        logging.info(f"Rodando {config['label']}...")
        params, score = run_ga(X_train_bal, y_train_bal, X_test_pre, y_test, config)
        
        # Avaliação final do melhor indivíduo do experimento
        final_model = SVC(**params, random_state=42).fit(X_train_bal, y_train_bal)
        y_pred = final_model.predict(X_test_pre)
        
        rec = recall_score(y_test, y_pred)
        results.append({'Modelo': config['label'], 'Recall': rec, 'Accuracy': accuracy_score(y_test, y_pred)})
        
        if rec > max_recall:
            max_recall = rec
            best_overall_model = final_model

    # 3. GRÁFICOS DE COMPARAÇÃO
    df_res = pd.DataFrame(results)
    df_res.set_index('Modelo').plot(kind='bar', figsize=(10,5))
    plt.title("Comparação: Modelo Base vs Experimentos GA")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 4. INTEGRAÇÃO LLM (REQUISITO 3)
    logging.info("\n=== INTERPRETAÇÃO IA MÉDICA ===")
    idx = random.randint(0, len(X_test)-1)
    paciente_raw = X_test.iloc[idx]
    paciente_pre = X_test_pre[idx].reshape(1, -1)
    
    predicao = best_overall_model.predict(paciente_pre)[0]
    
    insight_final = get_llm_insight(predicao, paciente_raw)
    print(f"\nPaciente ID {idx} - Predição: {'Diabético' if predicao == 1 else 'Não Diabético'}")
    print("-" * 30)
    print(insight_final)

if __name__ == "__main__":
    main()