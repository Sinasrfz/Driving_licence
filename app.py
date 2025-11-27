import streamlit as st
import pandas as pd
from gtts import gTTS
import base64
from io import BytesIO
import random

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_vocab():
    df = pd.read_excel("STEP4_vocab_with_examples.xlsx")   # your STEP4 final file
    df.fillna("", inplace=True)
    return df

df = load_vocab()


# ============================================================
# AUDIO GENERATOR
# ============================================================

def generate_audio(text):
    tts = gTTS(text, lang="it")
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    b64 = base64.b64encode(mp3_fp.read()).decode()
    return f'<audio autoplay="true" controls src="data:audio/mp3;base64,{b64}"></audio>'


# ============================================================
# MAIN SIDEBAR MENU
# ============================================================

st.sidebar.title("🇮🇹 Italian Learning App")
menu = st.sidebar.radio("Navigation", ["Dictionary", "Favourites", "Quiz"])


# ============================================================
# PAGE 1 — DICTIONARY VIEW
# ============================================================

def dictionary_view():

    st.title("📘 Italian Dictionary — Patente Vocabulary")

    # Sidebar search
    search = st.sidebar.text_input("Search word")

    df_filtered = df.copy()

    if search.strip():
        df_filtered = df_filtered[df_filtered["word"].str.contains(search, case=False)]

    if "dict_idx" not in st.session_state:
        st.session_state.dict_idx = 0

    words = df_filtered["word"].tolist()

    if len(words) == 0:
        st.info("No words found.")
        return

    # Protect index boundaries
    st.session_state.dict_idx = max(0, min(st.session_state.dict_idx, len(words)-1))

    selected_word = words[st.session_state.dict_idx]
    row = df_filtered[df_filtered["word"] == selected_word].iloc[0]

    # Navigation
    c1, c2, c3 = st.columns([1, 1, 5])

    with c1:
        if st.button("⬅ Previous"):
            if st.session_state.dict_idx > 0:
                st.session_state.dict_idx -= 1

    with c2:
        if st.button("Next ➡"):
            if st.session_state.dict_idx < len(words)-1:
                st.session_state.dict_idx += 1

    st.write(f"### 🔤 Word: **{selected_word}**")
    st.write(f"**English:** {row['translation']}")

    # Audio
    if st.button("🔊 Pronounce"):
        st.markdown(generate_audio(selected_word), unsafe_allow_html=True)

    # Examples
    st.write("### 📝 Example Sentences")
    st.write(row["example_sentence"] if row["example_sentence"] else "_No examples available._")

    # Favourites
    if "favourites" not in st.session_state:
        st.session_state.favourites = []

    if st.button("⭐ Add to Favourites"):
        if selected_word not in st.session_state.favourites:
            st.session_state.favourites.append(selected_word)
            st.success(f"Added **{selected_word}** to favourites!")


# ============================================================
# PAGE 2 — FAVOURITES
# ============================================================

def favourites_view():

    st.title("⭐ Favourite Words")

    favs = st.session_state.get("favourites", [])

    if not favs:
        st.info("No favourite words yet.")
        return

    for w in favs:
        row = df[df["word"] == w].iloc[0]
        st.write(f"### 🔤 {w}")
        st.write(f"**English:** {row['translation']}")
        st.write(f"**Examples:** {row['example_sentence']}")
        st.write("---")

    if st.button("❌ Clear All Favourites"):
        st.session_state.favourites = []
        st.warning("Favourites cleared.")


# ============================================================
# PAGE 3 — QUIZ MODE
# ============================================================

def quiz_view():

    st.title("🎯 Quiz Mode — Test Your Memory")

    if "quiz_word" not in st.session_state:
        st.session_state.quiz_word = random.choice(df["word"].tolist())

    quiz_word = st.session_state.quiz_word
    correct_answer = df[df["word"] == quiz_word]["translation"].iloc[0]

    st.write(f"### What is the meaning of: **{quiz_word}** ?")

    user_answer = st.text_input("Your answer:")

    if st.button("Check"):
        if user_answer.lower().strip() == correct_answer.lower().strip():
            st.success("Correct! 🎉")
        else:
            st.error(f"Incorrect ❌ — correct answer is: **{correct_answer}**")

    if st.button("Next Question"):
        st.session_state.quiz_word = random.choice(df["word"].tolist())
        st.experimental_rerun()


# ============================================================
# ROUTER
# ============================================================

if menu == "Dictionary":
    dictionary_view()

elif menu == "Favourites":
    favourites_view()

elif menu == "Quiz":
    quiz_view()
