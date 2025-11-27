# ============================================================
# app.py — Italian Patente Vocabulary Trainer (Streamlit)
# ============================================================

import os
import io
import random

import numpy as np
import pandas as pd
import streamlit as st
from gtts import gTTS


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="Italian Patente Vocab Trainer",
    layout="wide",
    page_icon="🚗",
)


# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------

@st.cache_data
def load_vocab():
    df4 = pd.read_excel("STEP4_vocab_with_examples.xlsx")

    cols = [c.lower() for c in df4.columns]
    df4.columns = cols

    if "word" not in df4.columns:
        raise ValueError("STEP4 file must contain a 'word' column.")
    if "english" not in df4.columns:
        df4["english"] = ""
    if "examples" not in df4.columns:
        df4["examples"] = ""

    df4 = df4[["word", "english", "examples"]]

    freq_path = "STEP2_clean_vocab.xlsx"
    if os.path.exists(freq_path):
        df2 = pd.read_excel(freq_path)
        if df2.shape[1] >= 2:
            df2 = df2.iloc[:, :2]
            df2.columns = ["word", "freq"]
        else:
            df2.columns = ["word"]
            df2["freq"] = np.nan
    else:
        df2 = pd.DataFrame({"word": df4["word"], "freq": np.nan})

    df = df4.merge(df2, on="word", how="left")

    df["word"] = df["word"].astype(str)
    df["english"] = df["english"].astype(str)
    df["examples"] = df["examples"].fillna("").astype(str)
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce")

    df["freq_rank"] = df["freq"].rank(method="dense", ascending=False)

    df = df.sort_values(["freq_rank", "word"], na_position="last").reset_index(drop=True)
    return df


# ------------------------------------------------------------
# FAVOURITES HANDLING
# ------------------------------------------------------------

FAV_FILE = "favourites.csv"


def load_favourites():
    if os.path.exists(FAV_FILE):
        try:
            df = pd.read_csv(FAV_FILE)
            return set(df["word"].astype(str).tolist())
        except Exception:
            return set()
    return set()


def save_favourites(favs: set):
    if len(favs) == 0:
        if os.path.exists(FAV_FILE):
            os.remove(FAV_FILE)
        return

    df = pd.DataFrame({"word": sorted(list(favs))})
    df.to_csv(FAV_FILE, index=False)


# ------------------------------------------------------------
# AUDIO (gTTS)
# ------------------------------------------------------------

@st.cache_data
def generate_tts_audio(word: str) -> bytes:
    tts = gTTS(text=word, lang="it")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ------------------------------------------------------------
# DICTIONARY VIEW (Word list removed)
# ------------------------------------------------------------

def dictionary_view(df: pd.DataFrame, fav_set: set):
    st.header("📘 Dictionary View")

    with st.sidebar:
        st.subheader("🔍 Filters")

        search = st.text_input("Search (Italian or English)", "").strip()
        starts_with = st.text_input("Starts with (Italian letter)", "").strip()
        only_favs = st.checkbox("⭐ Show favourites only", value=False)

        sort_by = st.selectbox(
            "Sort by",
            options=["Frequency (high → low)", "Italian (A → Z)", "English (A → Z)"],
        )

    df_filtered = df.copy()

    if search:
        s = search.lower()
        df_filtered = df_filtered[
            df_filtered["word"].str.lower().str.contains(s)
            | df_filtered["english"].str.lower().str.contains(s)
        ]

    if starts_with:
        sw = starts_with.lower()
        df_filtered = df_filtered[df_filtered["word"].str.lower().str.startswith(sw)]

    if only_favs:
        df_filtered = df_filtered[df_filtered["word"].isin(fav_set)]

    if sort_by == "Italian (A → Z)":
        df_filtered = df_filtered.sort_values("word")
    elif sort_by == "English (A → Z)":
        df_filtered = df_filtered.sort_values("english")
    else:
        df_filtered = df_filtered.sort_values(["freq_rank", "word"], na_position="last")

    col_left, col_detail = st.columns([1, 3])

    with col_left:
        st.subheader("🔤 Choose Word")

        if len(df_filtered) > 0:

            if "dict_selected_word" not in st.session_state:
                st.session_state.dict_selected_word = df_filtered["word"].iloc[0]

            words_list = df_filtered["word"].tolist()

            if st.session_state.dict_selected_word in words_list:
                current_index = words_list.index(st.session_state.dict_selected_word)
            else:
                current_index = 0

            selected_word = st.selectbox(
                "Select a word:",
                options=words_list,
                index=current_index,
                key="dict_word_select",
            )

            st.session_state.dict_selected_word = selected_word
        else:
            selected_word = None
            st.info("No words found.")

    with col_detail:
        st.subheader("🔍 Word Details")

        if selected_word is not None:
            row = df[df["word"] == selected_word].iloc[0]

            st.markdown(f"### 🇮🇹 **{row['word']}**")
            st.markdown(f"**🇬🇧 English:** {row['english'] or '*No translation*'}")

            if pd.notna(row["freq"]):
                st.markdown(f"- **Frequency in corpus:** {int(row['freq'])}")
            else:
                st.markdown("- **Frequency in corpus:** not available")

            if st.button("🔊 Hear pronunciation"):
                audio_bytes = generate_tts_audio(row["word"])
                st.audio(audio_bytes, format="audio/mp3")

            is_fav = row["word"] in fav_set
            fav1, fav2 = st.columns(2)

            with fav1:
                if not is_fav:
                    if st.button("⭐ Add to favourites"):
                        fav_set.add(row["word"])
                        save_favourites(fav_set)
                        st.success(f"Added '{row['word']}' to favourites.")
                else:
                    if st.button("🗑 Remove from favourites"):
                        fav_set.discard(row["word"])
                        save_favourites(fav_set)
                        st.warning(f"Removed '{row['word']}'.")

            st.markdown("---")
            st.markdown("### 📎 Example sentences")

            if row["examples"].strip():
                parts = [e.strip() for e in row["examples"].split("---") if e.strip()]
                for i, ex in enumerate(parts, start=1):
                    st.markdown(f"**{i}.** {ex}")
            else:
                st.info("No examples available.")


# ------------------------------------------------------------
# QUIZ VIEW
# ------------------------------------------------------------

def init_quiz_state(df: pd.DataFrame):
    if "quiz_active" not in st.session_state:
        st.session_state.quiz_active = False
    if "quiz_current" not in st.session_state:
        st.session_state.quiz_current = None
    if "quiz_options" not in st.session_state:
        st.session_state.quiz_options = []
    if "quiz_answer" not in st.session_state:
        st.session_state.quiz_answer = None
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
    if "quiz_total" not in st.session_state:
        st.session_state.quiz_total = 0
    if "quiz_pool" not in st.session_state:
        st.session_state.quiz_pool = df[df["english"].str.strip() != ""].copy()


def pick_new_question():
    pool = st.session_state.quiz_pool
    if pool.empty:
        return None, []

    idx = random.randint(0, len(pool) - 1)
    row = pool.iloc[idx]

    correct = row["english"]
    word = row["word"]

    others = pool[pool["word"] != word]
    distractors = (
        random.sample(others["english"].tolist(), 3)
        if len(others) >= 3
        else others["english"].tolist()
    )

    options = distractors + [correct]
    random.shuffle(options)

    return (word, correct), options


def quiz_view(df: pd.DataFrame):
    st.header("🧪 Quiz Mode — Italian → English")

    init_quiz_state(df)

    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.markdown("### 📊 Scoreboard")
        st.metric("Correct", st.session_state.quiz_score)
        st.metric("Total", st.session_state.quiz_total)

        if st.button("🔁 Reset"):
            st.session_state.quiz_score = 0
            st.session_state.quiz_total = 0
            st.session_state.quiz_current = None
            st.session_state.quiz_answer = None
            st.session_state.quiz_options = []
            st.success("Quiz reset!")

    with col_left:
        if st.session_state.quiz_current is None:
            if st.button("▶ Start Quiz"):
                qa, opts = pick_new_question()
                if qa:
                    st.session_state.quiz_current = qa
                    st.session_state.quiz_options = opts
        else:
            word, correct = st.session_state.quiz_current

            st.markdown(f"### 🇮🇹 What is the meaning of **'{word}'**?")

            selected = st.radio(
                "Choose:",
                st.session_state.quiz_options,
                key="quiz_select",
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ Check"):
                    st.session_state.quiz_answer = selected
                    st.session_state.quiz_total += 1
                    if selected == correct:
                        st.session_state.quiz_score += 1
                        st.success("Correct!")
                    else:
                        st.error(f"Incorrect — correct: **{correct}**")

            with col2:
                if st.button("⏭ Next"):
                    qa, opts = pick_new_question()
                    if qa:
                        st.session_state.quiz_current = qa
                        st.session_state.quiz_options = opts
                        st.session_state.quiz_answer = None

            st.markdown("---")
            st.markdown("### Example sentences")

            row = df[df["word"] == word].iloc[0]
            if row["examples"].strip():
                parts = [e.strip() for e in row["examples"].split("---") if e.strip()]
                for i, ex in enumerate(parts, start=1):
                    st.markdown(f"**{i}.** {ex}")
            else:
                st.info("No examples available.")



# ------------------------------------------------------------
# FLASHCARDS MODE (NEW)
# ------------------------------------------------------------

def init_flashcards(df):
    if "flash_index" not in st.session_state:
        st.session_state.flash_index = 0
    if "flash_flipped" not in st.session_state:
        st.session_state.flash_flipped = False
    if "flash_order" not in st.session_state:
        st.session_state.flash_order = df["word"].tolist()


def flashcards_view(df):
    st.header("🃏 Flashcards Mode — Learn by Flipping Cards")

    init_flashcards(df)

    words = st.session_state.flash_order
    idx = st.session_state.flash_index

    if idx < 0:
        st.session_state.flash_index = 0
        idx = 0
    if idx >= len(words):
        st.session_state.flash_index = len(words)-1
        idx = len(words)-1

    current_word = words[idx]
    row = df[df["word"] == current_word].iloc[0]

    st.markdown("---")

    # Card view
    if not st.session_state.flash_flipped:
        # FRONT
        st.markdown(
            f"""
            <div style='text-align:center; font-size:42px; font-weight:bold; padding:30px;'>
                {row['word']}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # BACK
        st.markdown(
            f"""
            <div style='font-size:26px; padding:20px;'>
                <b>Meaning:</b> {row['english']}<br><br>
                <b>Example:</b><br>
            </div>
            """,
            unsafe_allow_html=True
        )

        if row["examples"].strip():
            examples = row["examples"].split("---")
            for ex in examples[:2]:
                st.markdown(f"➡ {ex.strip()}")
        else:
            st.info("No example sentences found.")

    st.markdown("---")

    # Navigation Buttons
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.button("⬅ Previous"):
            st.session_state.flash_index -= 1
            st.session_state.flash_flipped = False

    with c2:
        if st.button("🔄 Flip"):
            st.session_state.flash_flipped = not st.session_state.flash_flipped

    with c3:
        if st.button("Next ➡"):
            st.session_state.flash_index += 1
            st.session_state.flash_flipped = False

    with c4:
        if st.button("🔀 Shuffle"):
            new_order = df.sample(frac=1)["word"].tolist()
            st.session_state.flash_order = new_order
            st.session_state.flash_index = 0
            st.session_state.flash_flipped = False
            st.success("Shuffled!")

    # Pronunciation
    if st.button("🔊 Hear pronunciation"):
        audio_bytes = generate_tts_audio(row["word"])
        st.audio(audio_bytes, format="audio/mp3")



# ------------------------------------------------------------
# MAIN PAGE (added project intro + author credit)
# ------------------------------------------------------------

def main():
    st.title("🚗 Italian Patente Vocabulary Trainer")

    # 📌 New introduction text
    st.markdown("""
This application is designed to help you learn and master the **Italian driver's licence vocabulary (Patente B)**  
using:

- 🇮🇹 Italian → 🇬🇧 English translations  
- 📝 Automatically extracted example sentences from real exam texts  
- 🔊 Pronunciation via text-to-speech  
- ⭐ Favourite lists  
- 🎯 Quiz-based learning  
- 🃏 Flashcards mode for fast memorisation  

---

**Developed by _Sina Sarfarazi_**  
📧 *sina.sarfarazi@unina.it*  
    """)

    st.markdown("---")

    try:
        df = load_vocab()
    except Exception as e:
        st.error(f"Error loading vocabulary files: {e}")
        st.stop()

    fav_set = load_favourites()

    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        mode = st.radio("Choose mode", ["Dictionary", "Quiz", "Flashcards"])

        st.markdown("---")
        st.markdown("**Data files:**")
        st.code("STEP4_vocab_with_examples.xlsx\nSTEP2_clean_vocab.xlsx")

    if mode == "Dictionary":
        dictionary_view(df, fav_set)
    elif mode == "Quiz":
        quiz_view(df)
    else:
        flashcards_view(df)


if __name__ == "__main__":
    main()
