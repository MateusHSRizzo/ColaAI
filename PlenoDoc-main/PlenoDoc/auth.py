import streamlit as st

# Credenciais para demonstração (NÃO USE EM PRODUÇÃO!)
# Apenas o usuário Administrador com a senha fornecida
USUARIOS = {
    "Administrador": "1234"
}

def pagina_login():
    """
    Exibe a interface de login com campos de usuário e senha.
    Após a validação, atualiza o estado da sessão para 'logged_in'.
    """
    st.title("ColaAI 😈")
    st.markdown("Faça login para acessar a aplicação.")
    
    with st.form("login_form"):
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        col1, col2 = st.columns([1, 2])
        with col1:
            login_button = st.form_submit_button("Entrar", use_container_width=True)

    if login_button:
        if username in USUARIOS and USUARIOS[username] == password:
            st.session_state.logged_in = True
            st.success("Login realizado com sucesso!")
            st.rerun()
        else:
            st.error("Usuário ou senha incorretos.")