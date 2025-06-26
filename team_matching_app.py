import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SW마에스트로 팀매칭", layout="centered")

st.title("🧑‍💻 SW마에스트로 AI 기반 팀매칭")
st.write("연수생 정보를 바탕으로 팀 구성 적합도를 분석합니다.")

# Sample student portfolio data
students = [
    {
        "name": "김지원",
        "skills": "Python TensorFlow Docker",
        "domain": "헬스케어 AI for Social Good",
        "style": "기술 집중형",
        "role": "AI 엔지니어"
    },
    {
        "name": "이민수",
        "skills": "React Node.js Firebase",
        "domain": "에듀테크 UX",
        "style": "커뮤니케이션 중심",
        "role": "PM"
    },
    {
        "name": "박하늘",
        "skills": "Python FastAPI MySQL",
        "domain": "헬스케어 데이터 분석",
        "style": "데이터 중심",
        "role": "백엔드"
    }
]

df = pd.DataFrame(students)
df['combined'] = df['skills'] + ' ' + df['domain'] + ' ' + df['style'] + ' ' + df['role']

# TF-IDF 기반 유사도 계산
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined'])
cos_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

st.subheader("📊 연수생 간 유사도 분석")
st.dataframe(pd.DataFrame(cos_sim_matrix, index=df['name'], columns=df['name']))

# 팀 조합 추천 예시
st.subheader("🤝 추천 팀 구성")

team1 = ("김지원", "박하늘", "이민수")
similarity_score = (
    cos_sim_matrix[0][1] + cos_sim_matrix[0][2] + cos_sim_matrix[1][2]
) / 3

st.markdown(f"""
**추천 팀 A**
- 👩 김지원 (AI 엔지니어)
- 👨 박하늘 (백엔드)
- 👨 이민수 (PM)
