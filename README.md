AS05
Feito por: Ana Beatriz Pessoa Braz
Descrição da tarefa
Este projeto apresenta um assistente conversacional gratuito baseado em Large Language Models (LLMs). Ele permite o upload de documentos em PDF, realiza a indexação do conteúdo por meio de embeddings e oferece respostas precisas a perguntas baseadas nas informações extraídas dos arquivos
Como rodar o projeto localmente?
# Clone o repositório
git clone https://github.com/biapessoab/as05.git

# Acesse a pasta do projeto
cd as05

# Crie um ambiente virtual (requer Python 3 instalado)
python -m venv venv

# Ative o ambiente virtual
# No Windows:
venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt

# Rode a aplicação com o Streamlit
streamlit run app.py
Como acessar o projeto hospedado?
Basta acessar: https://as05-anabeatrizbraz.streamlit.app/
