import streamlit as st
from components.utils import get_pg_connection  # assumes you have this defined

def login_page():
    st.subheader("🔐 使用者登入")
    with st.form("login_form"):
        username = st.text_input("使用者帳號 username")
        submit = st.form_submit_button("登入")

        if submit:
            with get_pg_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT user_id FROM users WHERE username = %s", (username,))
                    result = cur.fetchone()

                    if result:
                        user_id = result[0]
                        st.session_state.user = user_id
                        st.success("登入成功！")
                        st.rerun()
                    else:
                        st.error("無此使用者，請先註冊")
