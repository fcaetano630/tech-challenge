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
def train_random_forest(X_train, y_train, X_train_bal, y_train_bal, X_test, y_test):
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_smote = RandomForestClassifier(random_state=42)
    grid_rf = GridSearchCV(rf, param_grid_rf, scoring='recall', cv=cv, n_jobs=-1)
    grid_rf_smote = GridSearchCV(rf_smote, param_grid_rf, scoring='recall', cv=cv, n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    grid_rf_smote.fit(X_train_bal, y_train_bal)
    return grid_rf, grid_rf_smote

def train_logistic_regression(X_train, y_train, X_train_bal, y_train_bal, X_test, y_test):
    param_grid_lr = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'saga']
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    lr_smote = LogisticRegression(random_state=42, max_iter=1000)
    grid_lr = GridSearchCV(lr, param_grid_lr, scoring='recall', cv=cv, n_jobs=-1)
    grid_lr_smote = GridSearchCV(lr_smote, param_grid_lr, scoring='recall', cv=cv, n_jobs=-1)
    grid_lr.fit(X_train, y_train)
    grid_lr_smote.fit(X_train_bal, y_train_bal)
    return grid_lr, grid_lr_smote

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
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1]
}
def create_individual():
    return {k: random.choice(v) for k, v in HYPERPARAMS.items()}
def initialize_population(pop_size):
    return [create_individual() for _ in range(pop_size)]
def fitness_function(individual, X_train_bal, y_train_bal, X_test, y_test):
    warnings.filterwarnings('ignore')
    model = SVC(C=individual['C'], kernel=individual['kernel'], gamma=individual['gamma'], random_state=42)
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)
    return recall_score(y_test, y_pred, zero_division=0)
def select_parents(population, fitnesses, num_parents):
    parents = []
    for _ in range(num_parents):
        contenders = random.sample(list(zip(population, fitnesses)), 3)
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
    for gen in range(generations):
        fitnesses = [fitness_function(ind, X_train_bal, y_train_bal, X_test, y_test) for ind in population]
        idx = np.argmax(fitnesses)
        if fitnesses[idx] > best_fit:
            best_fit = fitnesses[idx]
            best_ind = population[idx]
        parents = select_parents(population, fitnesses, num_parents)
        next_pop = [best_ind]
        while len(next_pop) < pop_size:
            p1, p2 = random.sample(parents, 2)
            o1, o2 = crossover(p1, p2)
            next_pop.append(mutate(o1, mutation_rate))
            if len(next_pop) < pop_size:
                next_pop.append(mutate(o2, mutation_rate))
        population = next_pop
        print(f"Geração {gen+1}/{generations} - Melhor Recall: {best_fit:.4f} com {best_ind}")
    return best_ind, best_fit

# =====================
# 5. Execução Principal
# =====================
def main():

    # Integração simulada com LLM para explicação do diagnóstico
    def simulate_llm_response(outcome, features):
        explanation = f"O modelo previu que este paciente é **{outcome}**.\n\n"
        explanation += "Analisando as características fornecidas:\n"
        features_dict = features.iloc[0].to_dict()
        if outcome == 'Diabético':
            if 'Glucose' in features_dict and features_dict['Glucose'] > 120:
                explanation += f"  - A glicose ({features_dict['Glucose']:.1f}) está elevada, sendo um forte indicador.\n"
            if 'BMI' in features_dict and features_dict['BMI'] > 30:
                explanation += f"  - O IMC ({features_dict['BMI']:.1f}) indica sobrepeso/obesidade, um fator de risco.\n"
            if 'Age' in features_dict and features_dict['Age'] > 40:
                explanation += f"  - A idade ({features_dict['Age']}) pode aumentar a predisposição.\n"
            if 'DiabetesPedigreeFunction' in features_dict and features_dict['DiabetesPedigreeFunction'] > 0.5:
                explanation += f"  - O histórico familiar ({features_dict['DiabetesPedigreeFunction']:.2f}) também sugere maior risco.\n"
            explanation += "  - A combinação desses fatores contribuiu para o diagnóstico de diabetes."
        else:
            if 'Glucose' in features_dict and features_dict['Glucose'] < 100:
                explanation += f"  - Os níveis de glicose ({features_dict['Glucose']:.1f}) estão saudáveis, um bom sinal.\n"
            if 'BMI' in features_dict and features_dict['BMI'] < 25:
                explanation += f"  - O IMC ({features_dict['BMI']:.1f}) está na faixa normal, indicando menor risco.\n"
            explanation += "  - A maioria das características do paciente está dentro de faixas consideradas de baixo risco para diabetes pelo modelo."
        explanation += "\n\nEsta é uma explicação simulada. Em um cenário real com um LLM completo e acesso aos pesos/importâncias das features do modelo, a personalização e a profundidade da explicação seriam maiores."
        return explanation

        # ...existing code...

        # Exemplo de explicação em linguagem natural para um paciente de teste
        idx = random.randint(0, X_test.shape[0] - 1)
        paciente = X_test.iloc[[idx]]
        # Usa o melhor modelo geral (maior recall entre todos os modelos e experimentos)
        all_results = [r for r in results if r['Recall'] is not None]
        best_model_name = max(all_results, key=lambda x: x['Recall'])['Modelo']
        if 'GA Exp' in best_model_name:
            # Para simplificação, usa o modelo SVM (SMOTE) como exemplo
            model = grid_svm_smote.best_estimator_
        else:
            model = None
            for name, m in models:
                if name == best_model_name:
                    model = m
                    break
            if model is None:
                model = grid_svm_smote.best_estimator_
        # Prepara os dados do paciente para predição
        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), paciente.columns.tolist())
        ], remainder='drop')
        preprocessor.fit(X_train)
        paciente_pre_proc = preprocessor.transform(paciente)
        pred = model.predict(paciente_pre_proc)[0]
        outcome = 'Diabético' if pred == 1 else 'Não Diabético'
        logging.info("\n===============================")
        logging.info("🩺 Exemplo de explicação em linguagem natural para um paciente de teste:")
        logging.info(f"Características do paciente:\n{paciente.to_string(index=False)}")
        logging.info(f"Diagnóstico do modelo: {outcome}")
        logging.info(simulate_llm_response(outcome, paciente))
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
    grid_rf, grid_rf_smote = train_random_forest(X_train_pre, y_train, X_train_bal, y_train_bal, X_test_pre, y_test)
    grid_lr, grid_lr_smote = train_logistic_regression(X_train_pre, y_train, X_train_bal, y_train_bal, X_test_pre, y_test)
    grid_svm, grid_svm_smote = train_svm(X_train_pre, y_train, X_train_bal, y_train_bal, X_test_pre, y_test)

    # Avaliação
    results = []
    models = [
        ("Random Forest", grid_rf.best_estimator_),
        ("Random Forest (SMOTE)", grid_rf_smote.best_estimator_),
        ("Logistic Regression", grid_lr.best_estimator_),
        ("Logistic Regression (SMOTE)", grid_lr_smote.best_estimator_),
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
        {'pop_size': 10, 'generations': 5, 'mutation_rate': 0.1, 'num_parents': 5, 'label': 'GA Exp1 (pop=10, gen=5, mut=0.1)'},
        {'pop_size': 20, 'generations': 8, 'mutation_rate': 0.05, 'num_parents': 10, 'label': 'GA Exp2 (pop=20, gen=8, mut=0.05)'},
        {'pop_size': 8, 'generations': 12, 'mutation_rate': 0.2, 'num_parents': 4, 'label': 'GA Exp3 (pop=8, gen=12, mut=0.2)'}
    ]
    ga_results = []
    for exp in ga_experiments:
        logging.info(f"\n--- {exp['label']} ---")
        best_ind, best_fit = run_genetic_algorithm(
            X_train_bal, y_train_bal, X_test_pre, y_test,
            pop_size=exp['pop_size'],
            generations=exp['generations'],
            mutation_rate=exp['mutation_rate'],
            num_parents=exp['num_parents']
        )
        logging.info(f"{exp['label']} - Melhores hiperparâmetros: {best_ind}")
        logging.info(f"{exp['label']} - Melhor recall: {best_fit:.4f}")
        ga_results.append({
            'Modelo': exp['label'],
            'Acurácia': None,
            'Precisão': None,
            'Recall': best_fit,
            'F1': None,
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
    best_by_metric = df_bar.set_index('Modelo').idxmax()
    logging.info("\n🏆 Melhor modelo por métrica:")
    for metric in metrics:
        logging.info(f" - {metric}: {best_by_metric[metric]}")

if __name__ == "__main__":
    main()
