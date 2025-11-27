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
    """
    Load:
    - STEP4_vocab_with_examples.xlsx  (word, english, examples)
    - STEP2_clean_vocab.xlsx          (word, frequency)
    and merge them.
    """
    # Load main vocab with translations + examples
    df4 = pd.read_excel("STEP4_vocab_with_examples.xlsx")

    # Normalise column names
    cols = [c.lower() for c in df4.columns]
    df4.columns = cols

    # Expect columns: word, english, examples
    if "word" not in df4.columns:
        raise ValueError("STEP4 file must contain a 'word' column.")
    if "english" not in df4.columns:
        df4["english"] = ""
    if "examples" not in df4.columns:
        df4["examples"] = ""

    df4 = df4[["word", "english", "examples"]]

    # Load frequency file (if available)
    freq_path = "STEP2_clean_vocab.xlsx"
    if os.path.exists(freq_path):
        df2 = pd.read_excel(freq_path)
        # Use first two columns as word + freq
        if df2.shape[1] >= 2:
            df2 = df2.iloc[:, :2]
            df2.columns = ["word", "freq"]
        else:
            df2.columns = ["word"]
            df2["freq"] = np.nan
    else:
        df2 = pd.DataFrame({"word": df4["word"], "freq": np.nan})

    # Merge
    df = df4.merge(df2, on="word", how="left")

    # Clean up
    df["word"] = df["word"].astype(str)
    df["english"] = df["english"].astype(str)
    df["examples"] = df["examples"].fillna("").astype(str)
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce")

    # Frequency rank (higher freq = more important)
    df["freq_rank"] = df["freq"].rank(method="dense", ascending=False)

    # For convenience, keep a nice ordering
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
        # If empty, remove file if exists
        if os.path.exists(FAV_FILE):
            os.remove(FAV_FILE)
        return

    df = pd.DataFrame({"word": sorted(list(favs))})
    df.to_csv(FAV_FILE, index=False)


# ------------------------------------------------------------
# AUDIO (gTTS) — Pronunciation
# ------------------------------------------------------------

@st.cache_data
def generate_tts_audio(word: str) -> bytes:
    """
    Generate TTS audio (Italian) for a word and return raw bytes.
    Cached by Streamlit so repeated calls are fast.
    """
    tts = gTTS(text=word, lang="it")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ------------------------------------------------------------
# DICTIONARY VIEW
# ------------------------------------------------------------

def dictionary_view(df: pd.DataFrame, fav_set: set):
    st.header("📘 Dictionary View")

    # --- Sidebar filters ---
    with st.sidebar:
        st.subheader("🔍 Filters")

        search = st.text_input("Search (Italian or English)", "").strip()
        starts_with = st.text_input("Starts with (Italian letter)", "").strip()
        only_favs = st.checkbox("⭐ Show favourites only", value=False)

        sort_by = st.selectbox(
            "Sort by",
            options=["Frequency (high → low)", "Italian (A → Z)", "English (A → Z)"],
        )

    # --- Apply filters ---
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

    # Sorting
    if sort_by == "Italian (A → Z)":
        df_filtered = df_filtered.sort_values("word")
    elif sort_by == "English (A → Z)":
        df_filtered = df_filtered.sort_values("english")
    else:
        df_filtered = df_filtered.sort_values(["freq_rank", "word"], na_position="last")

    # --- Left: table; Right: detail panel ---
    col_table, col_detail = st.columns([2, 3])

    with col_table:
        st.subheader("📊 Word List")

        display_df = df_filtered[["word", "english", "freq"]].copy()
        display_df.rename(
            columns={"word": "Italian", "english": "English", "freq": "Frequency"},
            inplace=True,
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            height=500,
        )

        # ============================================================
        # *** FIXED SELECTBOX LOGIC (ONLY MODIFICATION MADE) ***
        # ============================================================
        if len(df_filtered) > 0:

            if "dict_selected_word" not in st.session_state:
                st.session_state.dict_selected_word = df_filtered["word"].iloc[0]

            words_list = df_filtered["word"].tolist()

            # compute the correct index
            if st.session_state.dict_selected_word in words_list:
                current_index = words_list.index(st.session_state.dict_selected_word)
            else:
                current_index = 0

            selected_word = st.selectbox(
                "Choose a word to inspect",
                options=words_list,
                index=current_index,
                key="dict_word_select",
            )

            # update persistent selection
            st.session_state.dict_selected_word = selected_word

        else:
            selected_word = None
            st.info("No words match the current filters.")
        # ============================================================

    with col_detail:
        st.subheader("🔍 Word Details")

        if selected_word is not None:
            row = df[df["word"] == selected_word].iloc[0]

            st.markdown(f"### 🇮🇹 **{row['word']}**")
            st.markdown(f"**🇬🇧 English:** {row['english'] or '*No translation*'}")

            # Frequency info
            if pd.notna(row["freq"]):
                st.markdown(f"- **Frequency in corpus:** {int(row['freq'])}")
            else:
                st.markdown("- **Frequency in corpus:** not available")

            # Audio
            if st.button("🔊 Hear pronunciation"):
                audio_bytes = generate_tts_audio(row["word"])
                st.audio(audio_bytes, format="audio/mp3")

            # Favourites
            is_fav = row["word"] in fav_set
            fav_col1, fav_col2 = st.columns(2)
            with fav_col1:
                if not is_fav:
                    if st.button("⭐ Add to favourites"):
                        fav_set.add(row["word"])
                        save_favourites(fav_set)
                        st.success(f"Added '{row['word']}' to favourites.")
                else:
                    if st.button("🗑 Remove from favourites"):
                        fav_set.discard(row["word"])
                        save_favourites(fav_set)
                        st.warning(f"Removed '{row['word']}' from favourites.")

            st.markdown("---")
            st.markdown("### 📎 Example sentences (from exam text)")

            if row["examples"].strip():
                parts = [e.strip() for e in row["examples"].split("---") if e.strip()]
                for i, ex in enumerate(parts, start=1):
                    st.markdown(f"**{i}.** {ex}")
            else:
                st.info("No example sentences found for this word.")


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

    correct_en = row["english"]
    italian_word = row["word"]

    others = pool[pool["word"] != italian_word]
    if len(others) >= 3:
        distractors = random.sample(others["english"].tolist(), 3)
    else:
        distractors = others["english"].tolist()

    options = distractors + [correct_en]
    random.shuffle(options)

    return (italian_word, correct_en), options


def quiz_view(df: pd.DataFrame):
    st.header("🧪 Quiz Mode — Italian → English")

    init_quiz_state(df)

    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.markdown("### 📊 Score")
        st.metric("Correct answers", st.session_state.quiz_score)
        st.metric("Total questions", st.session_state.quiz_total)

        if st.button("🔁 Reset quiz"):
            st.session_state.quiz_score = 0
            st.session_state.quiz_total = 0
            st.session_state.quiz_current = None
            st.session_state.quiz_options = []
            st.session_state.quiz_answer = None
            st.success("Quiz state reset.")

    with col_left:
        if st.session_state.quiz_current is None:
            if st.button("▶ Start quiz"):
                qa, opts = pick_new_question()
                if qa is None:
                    st.error("Not enough words to build a quiz.")
                else:
                    st.session_state.quiz_current = qa
                    st.session_state.quiz_options = opts
                    st.session_state.quiz_answer = None
        else:
            italian_word, correct_en = st.session_state.quiz_current

            st.markdown(f"### 🇮🇹 What is the meaning of **'{italian_word}'**?")

            selected_opt = st.radio(
                "Choose one:",
                options=st.session_state.quiz_options,
                index=0 if st.session_state.quiz_answer is None else
                st.session_state.quiz_options.index(st.session_state.quiz_answer),
                key="quiz_radio",
            )

            check_col, next_col = st.columns(2)

            with check_col:
                if st.button("✅ Check answer"):
                    st.session_state.quiz_answer = selected_opt
                    st.session_state.quiz_total += 1
                    if selected_opt == correct_en:
                        st.session_state.quiz_score += 1
                        st.success(f"Correct: '{italian_word}' = '{correct_en}'")
                    else:
                        st.error(
                            f"Wrong — Correct answer: '{correct_en}' "
                            f"(you chose '{selected_opt}')"
                        )

            with next_col:
                if st.button("⏭ Next question"):
                    qa, opts = pick_new_question()
                    if qa is None:
                        st.warning("No more questions available.")
                        st.session_state.quiz_current = None
                        st.session_state.quiz_options = []
                    else:
                        st.session_state.quiz_current = qa
                        st.session_state.quiz_options = opts
                        st.session_state.quiz_answer = None

            st.markdown("---")
            st.markdown("### 📎 Example sentences for this word")
            row = df[df["word"] == italian_word].iloc[0]
            if row["examples"].strip():
                parts = [e.strip() for e in row["examples"].split("---") if e.strip()]
                for i, ex in enumerate(parts, start=1):
                    st.markdown(f"**{i}.** {ex}")
            else:
                st.info("No example sentences available.")


# ------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------

def main():
    st.title("🚗 Italian Patente Vocabulary Trainer")

    try:
        df = load_vocab()
    except Exception as e:
        st.error(f"Error loading vocabulary files: {e}")
        st.stop()

    fav_set = load_favourites()

    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        mode = st.radio("Choose mode", options=["Dictionary", "Quiz"])

        st.markdown("---")
        st.markdown("**Data files used:**")
        st.code("STEP4_vocab_with_examples.xlsx\nSTEP2_clean_vocab.xlsx")

    if mode == "Dictionary":
        dictionary_view(df, fav_set)
    else:
        quiz_view(df)


if __name__ == "__main__":
    main()
