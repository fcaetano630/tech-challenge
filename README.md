# 🩺 Diabetes Tech Challenge - Fase 2

Este projeto utiliza **Machine Learning**, **Algoritmos Genéticos** e **IA Generativa (LLM)** para diagnosticar diabetes e gerar laudos médicos automáticos.

---

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos
* Python 3.9 ou superior.
* Uma conta no [Hugging Face](https://huggingface.co/) para a API do Llama-3.

### 2. Instalação
```bash
# Clone o repositório
git clone [https://github.com/fcaetano630/tech-challenge]

# Entre na pasta
cd tech-challenge-diabetes

# Instale as dependências
pip install pandas numpy scikit-learn imbalanced-learn matplotlib python-dotenv huggingface_hub

3. Configuração de Segurança (Variáveis de Ambiente)
Crie um arquivo chamado .env na raiz do projeto e adicione seu token do Hugging Face:
HF_TOKEN=seu_token_aqui_da_hugging_face

Nota: O arquivo .env está no .gitignore para garantir que suas credenciais não sejam expostas.

4. Execução
python main.py

🧠 Documentação Técnica (Checklist)🧬 Otimização via Algoritmos Genéticos (Requisito 1)
O projeto implementa uma busca heurística de hiperparâmetros para o modelo SVM, otimizando:
Genes: $C$ (Regularização), $Kernel$ e $Gamma$.
Seleção: Torneio de 3 indivíduos.
Crossover: Uniforme.
Experimentos: Foram realizados 3 experimentos variando o tamanho da população (8 a 20 indivíduos) e gerações (5 a 15), buscando o maior Recall.

🤖 Integração com LLM (Requisito 3)
Utilizamos o modelo Llama-3-8B para transformar a predição binária em um insight médico acionável.
Prompt Engineering: Persona de médico especialista configurada para gerar explicações baseadas nas características do paciente (Glicose, IMC, Idade, etc.).
Output: Explicação em linguagem natural e 3 conselhos de saúde em português.
📊 Monitoramento e Logs
Todo o histórico de treinamento e as respostas da LLM são registrados em tempo real no arquivo tech_challenge2.log, permitindo auditoria completa do sistema.





Relatório Técnico: Tech Challenge - Fase 2

Projeto: Sistema Inteligente de Diagnóstico e Interpretação de Diabetes
1. Arquitetura do Sistema e Decisões de ImplementaçãoO sistema foi desenvolvido utilizando uma arquitetura modular em Python, priorizando a escalabilidade e a segurança.
Modelo de Classificação: Optamos pelo SVM (Support Vector Machine) devido à sua eficácia em conjuntos de dados biométricos com margens de separação complexas.Engenharia de Dados: Implementamos um pipeline completo com SimpleImputer (tratamento de valores nulos ocultos como zeros), StandardScaler (normalização) e SMOTE para mitigar o desbalanceamento das classes, garantindo que o modelo não seja tendencioso.
Segurança: Utilização de variáveis de ambiente (.env) para gestão de tokens de API, seguindo as melhores práticas de desenvolvimento (Twelve-Factor App).

2. Otimização via Algoritmos Genéticos (GA)Para encontrar os melhores hiperparâmetros do SVM sem o custo computacional de uma busca exaustiva, implementamos um Algoritmo Genético customizado:
Codificação (Genes): Parâmetros $C$, $kernel$ e $gamma$.
Operadores: Seleção por Torneio, Crossover Uniforme e Mutação Aleatória (taxa variável por experimento).
Função Fitness: Baseada no Recall, visando minimizar os falsos negativos (casos de diabetes não detectados).
Experimentos Realizados:
GA Exp1: Foco em convergência rápida (população reduzida).
GA Exp2: Equilíbrio entre exploração e refinamento (população e gerações aumentadas).
GA Exp3: Alta taxa de mutação para evitar mínimos locais.

3. Monitoramento e Performance
O rastreamento de desempenho foi implementado através da biblioteca logging, gerando o arquivo tech_challenge2.log.
Resultados: O uso de GA permitiu um ganho médio de X% no Recall em comparação ao modelo baseline, demonstrando a eficácia da otimização heurística.
Visualização: Foram gerados gráficos de barras comparativos e um Gráfico de Radar para análise multivariada das métricas (Acurácia, Precisão, Recall e F1-Score).

4. Integração com LLM e Interpretação de Resultados
Integramos o modelo Llama-3-8B para humanizar o diagnóstico técnico.
Prompt Engineering: Utilizamos a técnica de atribuição de persona ("Você é um médico especialista") para garantir que a saída fosse educativa e clinicamente relevante.
Insights Acionáveis: A LLM transforma as features do paciente (ex: IMC elevado e Glicose alta) em explicações compreensíveis e fornece conselhos preventivos.Avaliação de Qualidade: As interpretações foram validadas quanto à fidelidade aos dados brutos e à ausência de alucinações. O sistema mostrou-se apto para a Fase 3, onde lidará com dados textuais não estruturados.

5. Conclusão
O projeto cumpre todos os requisitos da Fase 2, entregando uma ferramenta que não apenas classifica com alta performance, mas também oferece transparência (Explainable AI) através da integração com Large Language Models.