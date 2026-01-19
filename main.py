# main.py
"""
Script principal para classificação de diabetes com Random Forest, SVM, Regressão Logística e otimização via Algoritmo Genético.
Organizado em funções reutilizáveis e pronto para execução local.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import random
import warnings
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# =====================
# 0. Configuração LLM (Hugging Face)
# =====================
def initialize_llm():
    """Inicializa o cliente LLM com token do .env"""
    path_do_script = Path(__file__).parent.absolute()
    caminho_env = path_do_script / ".env"
    load_dotenv(dotenv_path=caminho_env)
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    if HF_TOKEN:
        logging.info(f"✅ Token LLM carregado: {HF_TOKEN[:8]}...")
        return InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=HF_TOKEN)
    else:
        logging.warning(f"⚠️  Token LLM não encontrado em {caminho_env}")
        return None

def get_llm_insight(client, outcome, raw_features):
    """Gera explicação usando Llama-3 para classificação de diabetes"""
    if client is None:
        return "Insight LLM indisponível (token não configurado)"
    
    status = "Diabético" if outcome == 1 else "Não Diabético"
    features_dict = raw_features.to_dict() if hasattr(raw_features, 'to_dict') else raw_features
    
    messages = [
        {
            "role": "user",
            "content": f"Você é um médico especialista em diabetes. Explique por que um paciente com esses dados: {features_dict} foi classificado como {status}. Dê 3 conselhos de saúde em português."
        }
    ]
    
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=500,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro na conexão LLM: {str(e)[:100]}"

# =====================
# 1. Carregamento e Preprocessamento
# =====================
def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    cols_with_invalid_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_with_invalid_zero:
        df[col] = df[col].replace(0, np.nan)
    target_col = "Outcome"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols)
    ], remainder='drop')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_pre, y_train)
    return X_train, X_test, y_train, y_test, X_train_pre, X_test_pre, X_train_bal, y_train_bal, X, df

# =====================
# 2. Treinamento dos Modelos
# =====================

def train_svm(X_train, y_train, X_train_bal, y_train_bal, X_test, y_test):
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm = SVC(class_weight='balanced', probability=True, random_state=42)
    svm_smote = SVC(probability=True, random_state=42)
    grid_svm = GridSearchCV(svm, param_grid, scoring='recall', cv=cv, n_jobs=-1)
    grid_svm_smote = GridSearchCV(svm_smote, param_grid, scoring='recall', cv=cv, n_jobs=-1)
    grid_svm.fit(X_train, y_train)
    grid_svm_smote.fit(X_train_bal, y_train_bal)
    return grid_svm, grid_svm_smote

# =====================
# 3. Avaliação dos Modelos
# =====================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'report': classification_report(y_test, y_pred, zero_division=0)
    }

# =====================
# 4. Algoritmo Genético para SVM
# =====================
HYPERPARAMS = {
    'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100]
}
def create_individual():
    return {k: random.choice(v) for k, v in HYPERPARAMS.items()}
def initialize_population(pop_size):
    return [create_individual() for _ in range(pop_size)]
def fitness_function(individual, X_train_bal, y_train_bal, X_test, y_test):
    warnings.filterwarnings('ignore')
    try:
        model = SVC(C=individual['C'], kernel=individual['kernel'], gamma=individual['gamma'], random_state=42, max_iter=10000)
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        return recall, f1, accuracy, precision
    except Exception as e:
        # Return worst case if model fails
        print(f"  Aviso: Modelo falhou com {individual}: {str(e)[:50]}")
        return 0.0, 0.0, 0.0, 0.0
def select_parents(population, fitnesses, num_parents):
    parents = []
    # Handle case where we can't sample 3 contenders
    sample_size = min(3, len(population))
    for _ in range(num_parents):
        contenders = random.sample(list(zip(population, fitnesses)), sample_size)
        parents.append(max(contenders, key=lambda x: x[1])[0])
    return parents
def crossover(parent1, parent2):
    o1, o2 = {}, {}
    for k in HYPERPARAMS:
        if random.random() < 0.5:
            o1[k], o2[k] = parent1[k], parent2[k]
        else:
            o1[k], o2[k] = parent2[k], parent1[k]
    return o1, o2
def mutate(individual, mutation_rate):
    for k, v in HYPERPARAMS.items():
        if random.random() < mutation_rate:
            individual[k] = random.choice(v)
    return individual
def run_genetic_algorithm(X_train_bal, y_train_bal, X_test, y_test, pop_size=10, generations=5, mutation_rate=0.1, num_parents=5):
    population = initialize_population(pop_size)
    best_ind, best_fit = None, -1
    best_metrics = None
    for gen in range(generations):
        print(f"Geração {gen+1}/{generations} - Avaliando população de {len(population)} indivíduos...")
        import sys
        sys.stdout.flush()
        
        for i, ind in enumerate(population):
            print(f"  Avaliando indivíduo {i+1}/{len(population)}...", end='\r')
            sys.stdout.flush()
            metrics = fitness_function(ind, X_train_bal, y_train_bal, X_test, y_test)
            # Handle the case where metrics might be incomplete
            if len(metrics) == 4:
                fitnesses_data = [metrics if ind == ind else m for m in [metrics]]
            else:
                fitnesses_data.append(metrics)
        
        # Recalculate fitnesses_data properly
        fitnesses_data = []
        for ind in population:
            metrics = fitness_function(ind, X_train_bal, y_train_bal, X_test, y_test)
            fitnesses_data.append(metrics)
        
        fitnesses = [f[0] for f in fitnesses_data]  # Extract recall for comparison
        idx = np.argmax(fitnesses)
        if fitnesses[idx] > best_fit:
            best_fit = fitnesses[idx]
            best_ind = population[idx]
            best_metrics = fitnesses_data[idx]  # Store all metrics (recall, f1, accuracy, precision)
        
        # Ensure we have enough parents for crossover
        num_parents_actual = max(2, min(num_parents, len(population)))
        parents = select_parents(population, fitnesses, num_parents_actual)
        
        next_pop = [best_ind]
        while len(next_pop) < pop_size:
            # Ensure we always have at least 2 parents to sample from
            if len(parents) >= 2:
                p1, p2 = random.sample(parents, 2)
            else:
                p1 = parents[0] if parents else best_ind
                p2 = parents[0] if parents else best_ind
            
            o1, o2 = crossover(p1, p2)
            next_pop.append(mutate(o1, mutation_rate))
            if len(next_pop) < pop_size:
                next_pop.append(mutate(o2, mutation_rate))
        population = next_pop
        print(f"Geração {gen+1}/{generations} - Melhor Recall: {best_fit:.4f}              ")
        sys.stdout.flush()
    
    # Ensure best_metrics is not None before unpacking
    if best_metrics is None:
        best_metrics = (best_fit, 0, 0, 0)
    
    return best_ind, best_fit, best_metrics[0], best_metrics[1], best_metrics[2], best_metrics[3]

# =====================
# 5. Execução Principal
# =====================
def main():
    # Configuração de logging
    logging.basicConfig(
        filename='tech_challenge2.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    csv_path = "diabetes.csv"  # Ajuste o caminho conforme necessário
    X_train, X_test, y_train, y_test, X_train_pre, X_test_pre, X_train_bal, y_train_bal, X, df = load_and_preprocess_data(csv_path)
    logging.info("Dados carregados e pré-processados.")

    # Treinamento dos modelos
    grid_svm, grid_svm_smote = train_svm(X_train_pre, y_train, X_train_bal, y_train_bal, X_test_pre, y_test)

    # Avaliação
    results = []
    models = [
        ("SVM", grid_svm.best_estimator_),
        ("SVM (SMOTE)", grid_svm_smote.best_estimator_)
    ]
    logging.info("\n===============================")
    logging.info("📊 Comparativo de Modelos")
    logging.info("===============================")
    for name, model in models:
        metrics = evaluate_model(model, X_test_pre, y_test)
        results.append({
            'Modelo': name,
            'Acurácia': metrics['accuracy'],
            'Precisão': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'Relatório': metrics['report']
        })
        logging.info(f"\n{name}:")
        logging.info(metrics)

    # Algoritmo Genético para SVM - Três experimentos
    logging.info("\nOtimização SVM com Algoritmo Genético (3 experimentos):")
    ga_experiments = [
        {'pop_size': 10, 'generations': 20, 'mutation_rate': 0.1, 'num_parents': 5, 'label': 'GA Exp1 (pop=10, gen=20, mut=0.1)'},
        {'pop_size': 20, 'generations': 32, 'mutation_rate': 0.05, 'num_parents': 10, 'label': 'GA Exp2 (pop=20, gen=32, mut=0.05)'},
        {'pop_size': 8, 'generations': 48, 'mutation_rate': 0.2, 'num_parents': 4, 'label': 'GA Exp3 (pop=8, gen=48, mut=0.2)'}
    ]
    ga_results = []
    for exp in ga_experiments:
        logging.info(f"\n--- {exp['label']} ---")
        best_ind, best_fit, recall, f1, accuracy, precision = run_genetic_algorithm(
            X_train_bal, y_train_bal, X_test_pre, y_test,
            pop_size=exp['pop_size'],
            generations=exp['generations'],
            mutation_rate=exp['mutation_rate'],
            num_parents=exp['num_parents']
        )
        logging.info(f"{exp['label']} - Melhores hiperparâmetros: {best_ind}")
        logging.info(f"{exp['label']} - Melhor recall: {recall:.4f}")
        logging.info(f"{exp['label']} - Melhor acurácia: {accuracy:.4f}")
        logging.info(f"{exp['label']} - Melhor precisão: {precision:.4f}")
        logging.info(f"{exp['label']} - Melhor f1: {f1:.4f}")
        ga_results.append({
            'Modelo': exp['label'],
            'Acurácia': accuracy,
            'Precisão': precision,
            'Recall': recall,
            'F1': f1,
            'Relatório': None
        })

    # Adiciona resultados do GA ao comparativo
    results.extend(ga_results)

    # Geração dos gráficos
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df_results = pd.DataFrame(results)
    # Remove modelos sem métricas completas para gráficos de barras
    df_bar = df_results.dropna(subset=['Acurácia', 'Precisão', 'Recall', 'F1'])
    metrics = ['Acurácia', 'Precisão', 'Recall', 'F1']

    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        plt.barh(df_bar['Modelo'], df_bar[metric], color=['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2', '#edc948'])
        plt.title(metric)
        plt.xlabel('Score')
        plt.xlim(0, 1)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Gráfico de radar
    df_radar = df_bar.set_index('Modelo')
    labels = metrics
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for idx, row in df_radar.iterrows():
        values = row[labels].tolist()
        values += values[:1]
        ax.plot(angles, values, label=idx)
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Radar Chart - Comparação de Modelos", size=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()

    # Exibe o melhor modelo por métrica
    best_by_metric = df_bar.set_index('Modelo')[metrics].idxmax()
    logging.info("\n🏆 Melhor modelo por métrica:")
    for metric in metrics:
        logging.info(f" - {metric}: {best_by_metric[metric]}")

    # Integração LLM para explicação de resultado (REQUISITO 3)
    llm_client = initialize_llm()
    if llm_client:
        logging.info("\n" + "="*50)
        logging.info("🤖 INTERPRETAÇÃO IA (LLAMA-3)")
        logging.info("="*50)
        
        # Seleciona um paciente aleatório para explicação
        idx = random.randint(0, X_test.shape[0] - 1)
        paciente_raw = X_test.iloc[idx]
        paciente_pre = X_test_pre[idx].reshape(1, -1)
        
        # Usa o melhor modelo por recall
        best_recall_model_name = best_by_metric['Recall']
        best_model = None
        
        # Busca o modelo correspondente
        for name, m in models:
            if name == best_recall_model_name:
                best_model = m
                break
        
        if best_model is None:
            best_model = grid_svm_smote.best_estimator_
        
        # Faz predição
        predicao = best_model.predict(paciente_pre)[0]
        status = "Diabético" if predicao == 1 else "Não Diabético"
        
        logging.info(f"\nPaciente #{idx} - Diagnóstico: {status}")
        logging.info(f"Características do paciente:")
        logging.info(f"{paciente_raw.to_string()}")
        logging.info(f"\nExplicação IA (Llama-3):")
        logging.info("-" * 50)
        
        # Obtém insight do LLM
        insight = get_llm_insight(llm_client, predicao, paciente_raw)
        logging.info(insight)
        logging.info("-" * 50)

if __name__ == "__main__":
    main()
