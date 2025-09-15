import streamlit as st
import os
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Importa as funções dos outros arquivos
from auth import pagina_login
from data_processing import atualizar_vetores, inicializar_retriever, carregar_documentos

# --- Constantes e Modelos de IA ---
CAMINHO_DOCUMENTOS = "dados_docs"
MODELOS_DISPONIVEIS = {
    'Groq (Limitado)': {'versao_api': ['openai/gpt-oss-120b'], 'chat': ChatGroq},
    'OpenAI (Premium)': {'versao_api': ['gpt-4o-mini'], 'chat': ChatOpenAI}
}

# --- Inicialização do Estado da Sessão ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'memoria' not in st.session_state: st.session_state.memoria = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
if 'chain' not in st.session_state: st.session_state.chain = None
if 'retriever' not in st.session_state: st.session_state.retriever = None

# --- Funções da Interface (UI) ---
def pagina_chat():
    """Renderiza a página principal do chat."""
    st.title("Bem-vindo ao ColaAI 😈")
    st.write("Seu amiguinho na hora da prova.")
    st.divider()
    
    # Exibe o histórico da conversa
    for message in st.session_state.memoria.chat_memory.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)
    
    if not st.session_state.chain:
        st.warning("⚠️ O modelo não foi inicializado. Configure-o na barra lateral.")
        st.chat_input('Inicialize o modelo para começar a conversar.', disabled=True)
        return
    
    if input_usuario := st.chat_input('Faça sua pergunta ao PlenoDoc...'):
        with st.chat_message("user"):
            st.markdown(input_usuario)
        
        with st.spinner("Analisando documentos e pensando..."):
            try:
                chat_history = st.session_state.memoria.load_memory_variables({})['chat_history']
                resposta = st.session_state.chain.invoke({"input": input_usuario, "chat_history": chat_history})
                
                st.session_state.memoria.save_context({"input": input_usuario}, {"output": resposta["answer"]})
                
                with st.chat_message("ai"):
                    st.markdown(resposta["answer"])
            except Exception as e:
                st.error(f"❌ Erro ao processar a consulta: {e}")

def painel_documentos():
    """Painel interativo para upload e gerenciamento de documentos."""
    st.subheader("Adicionar Novos Documentos")
    arquivos = st.file_uploader(
        'Arraste e solte arquivos aqui', 
        accept_multiple_files=True, 
        type=['pdf', 'csv', 'txt', 'docx'],
        label_visibility="collapsed"
    )
    if st.button("Processar Arquivos", use_container_width=True):
        if arquivos:
            caminhos_arquivos = []
            os.makedirs(CAMINHO_DOCUMENTOS, exist_ok=True)
            for arquivo in arquivos:
                caminho_salvar = os.path.join(CAMINHO_DOCUMENTOS, arquivo.name)
                with open(caminho_salvar, 'wb') as f: f.write(arquivo.getbuffer())
                caminhos_arquivos.append(caminho_salvar)
            with st.spinner("Atualizando base de conhecimento..."):
                documentos = carregar_documentos(caminhos_arquivos)
                atualizar_vetores(documentos)
                st.rerun()
        else: st.warning("Nenhum arquivo selecionado para processar.")

    st.divider()

    st.subheader("Documentos na Base")
    if not os.path.exists(CAMINHO_DOCUMENTOS) or not os.listdir(CAMINHO_DOCUMENTOS):
        st.info("Nenhum documento encontrado.")
    else:
        for nome_arquivo in os.listdir(CAMINHO_DOCUMENTOS):
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.text(nome_arquivo)
            with col2:
                # Botão de remoção com ícone de lixeira
                if st.button("🗑️", key=f"remover_{nome_arquivo}", use_container_width=True, help=f"Remover {nome_arquivo}"):
                    caminho_arquivo = os.path.join(CAMINHO_DOCUMENTOS, nome_arquivo)
                    os.remove(caminho_arquivo)
                    with st.spinner(f"Removendo '{nome_arquivo}'..."):
                        documentos_restantes = carregar_documentos([os.path.join(CAMINHO_DOCUMENTOS, f) for f in os.listdir(CAMINHO_DOCUMENTOS)])
                        atualizar_vetores(documentos_restantes)
                        st.toast(f"'{nome_arquivo}' removido com sucesso.", icon="✅")
                        st.rerun()

def sidebar():
    """Renderiza a barra lateral de configurações."""
    st.sidebar.header("Configurações do ColaAI")
    
    tabs = st.sidebar.tabs(['Gerenciar Documentos', 'Seleção de Modelo'])
    
    with tabs[0]:
        painel_documentos()
    
    with tabs[1]:
        st.subheader("Modelo de IA")
        selecao_provedor = st.selectbox('Provedor', MODELOS_DISPONIVEIS.keys(), label_visibility="collapsed")
        modelo = st.selectbox('Modelo', MODELOS_DISPONIVEIS[selecao_provedor]['versao_api'], label_visibility="collapsed")
        
        st.subheader("Chave de API")
        api_key = st.text_input(f'Chave da API para {selecao_provedor}', type='password', label_visibility="collapsed")
        
        if st.button('Inicializar PlenoDoc', use_container_width=True, type="primary"):
            if not api_key.strip():
                st.error('É necessário fornecer uma chave de API válida.')
            else:
                inicializar_modelo(selecao_provedor, modelo, api_key.strip())
    
    # Botão de Logout posicionado no final para melhor visualização
    st.sidebar.divider()
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.memoria.clear()
        st.session_state.chain = None
        st.session_state.retriever = None
        st.rerun()


def inicializar_modelo(selecao_provedor, modelo, api_key):
    """Inicializa o LLM e a cadeia de conversação RAG."""
    retriever = inicializar_retriever()
    if not retriever:
        st.error("A base de conhecimento (vetores) não está pronta.")
        return

    try:
        llm = MODELOS_DISPONIVEIS[selecao_provedor]['chat'](model=modelo, api_key=api_key, temperature=0.3)
        prompt_historico = ChatPromptTemplate.from_messages([
            ("system", "Com base na conversa abaixo, gere uma pergunta de busca que possa ser entendida sem o histórico do chat."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt_historico)
        prompt_resposta = ChatPromptTemplate.from_messages([
            ("system", """Você é um consultor especialista chamado PlenoDoc. Sua missão é responder às perguntas do usuário de forma detalhada e precisa.
REGRAS:
**1. PERSONA E IDENTIDADE:**
A partir de agora, você atuará como o **"ColaAI"**. Você é um assistente de conhecimento inteligente, amigável e prestativo. Sua personalidade é a de um parceiro de estudos ou um colega experiente, sempre disposto a ajudar a entender um assunto em profundidade. Seu tom é conversacional, mas preciso.

**2. OBJETIVO PRINCIPAL:**
Seu objetivo é ajudar o usuário a compreender e explorar os tópicos presentes nos documentos fornecidos. Você deve usar os documentos como a base principal da conversa, mas enriquecer as respostas com seu conhecimento geral para fornecer explicações completas, exemplos práticos e conexões com outros assuntos relevantes.

**3. REGRAS DE OPERAÇÃO:**

* **REGRA 1: O CONTEXTO É A BASE (MAS NÃO A JAULA):** A sua fonte primária de verdade e o ponto de partida para qualquer resposta é o contexto dos documentos fornecidos. Sempre priorize a informação contida neles.

* **REGRA 2: ENRIQUEÇA E EXPANDA:** Diferente de um assistente restrito, você **deve** usar seu conhecimento prévio para melhorar a resposta. Se o documento cita um termo técnico, explique-o de forma simples. Se menciona um evento, forneça o contexto histórico. Se apresenta uma fórmula, dê um exemplo prático de seu uso. O objetivo é entregar uma resposta mais completa e útil do que o documento sozinho poderia oferecer.

* **REGRA 3: O FILTRO DE RELEVÂNCIA (A REGRA DOS 70%):** Você deve manter o foco no universo temático dos documentos.
    * Se a pergunta do usuário tiver uma **conexão substancial com o contexto** (aproximadamente 70% ou mais de sobreposição temática), responda-a generosamente, combinando as informações dos documentos com seu conhecimento geral.
    * Se a pergunta for **muito distante ou completamente fora do escopo** do material fornecido (ex: perguntar sobre culinária quando o documento é sobre programação), informe educadamente que o assunto foge do material de estudo atual, mas se coloque à disposição para responder a perguntas relacionadas ao contexto.

* **REGRA 4: SEJA UM BOM CONVERSADOR:** Mantenha um diálogo natural. Faça perguntas de esclarecimento se necessário e use uma linguagem acessível, evitando jargões sempre que possível ou explicando-os quando inevitável.
[CONTEXTO]
{context}
[/CONTEXTO]"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        document_chain = create_stuff_documents_chain(llm, prompt_resposta)
        st.session_state.chain = create_retrieval_chain(retriever_chain, document_chain)
        st.success(f"✅ Modelo {modelo} inicializado com sucesso!")
    except Exception as e:
        st.error(f"❌ Falha ao inicializar o modelo: {e}")
        st.session_state.chain = None

# --- Função Principal que Executa a Aplicação ---
def main():
    if st.session_state.logged_in:
        sidebar()
        pagina_chat()
    else:
        pagina_login()
        
if __name__ == '__main__':

    main()
