import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
from collections import defaultdict
import nltk
import warnings
import json
import os 

warnings.filterwarnings('ignore')

# --- Performance Caching for Cloud Deployment ---
@st.cache_resource
def load_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    return True

@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load dependencies
nltk_loaded = load_nltk_data()
stop_words = set(nltk.corpus.stopwords.words('english'))

# --- Custom JSON Encoder ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class TalentSightAI:
    def __init__(self):
        with st.spinner('🔧 Loading TalentSight AI models... This might take a minute on first run'):
            self.semantic_model = load_semantic_model()
            self.stop_words = stop_words
        
    def extract_job_requirements(self, job_description):
        requirements = {
            'skills': [], 'experience_years': 0, 'education': [], 
            'soft_skills': [], 'seniority_level': ''
        }
        
        # Experience
        exp_patterns = [r'(\d+)\+?\s*(?:years|yrs).*?(?:experience|exp)', r'(?:experience|exp).*?(\d+)\+?\s*(?:years|yrs)']
        for pattern in exp_patterns:
            match = re.search(pattern, job_description.lower())
            if match:
                requirements['experience_years'] = int(match.group(1))
                break
        
        # Skills
        skill_keywords = ['python', 'java', 'javascript', 'react', 'angular', 'vue', 'node', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'ci/cd', 'git', 'rest api', 'graphql', 'machine learning', 'deep learning', 'nlp', 'computer vision', 'ai', 'data analysis', 'statistics', 'tableau', 'power bi', 'excel', 'project management', 'agile', 'scrum', 'kanban', 'leadership', 'communication', 'teamwork', 'problem solving', 'critical thinking']
        job_lower = job_description.lower()
        requirements['skills'] = [skill for skill in skill_keywords if skill in job_lower]
        
        # Seniority
        if any(word in job_lower for word in ['senior', 'lead', 'principal', 'architect']):
            requirements['seniority_level'] = 'Senior'
        elif any(word in job_lower for word in ['junior', 'associate', 'entry']):
            requirements['seniority_level'] = 'Junior'
        else:
            requirements['seniority_level'] = 'Mid-level'
        return requirements
    
    def analyze_candidate_profile(self, candidate_data):
        profile = {'career_progression': 0, 'stability_score': 0, 'recent_activity_score': 0}
        if 'career_history' in candidate_data:
            roles = str(candidate_data['career_history']).split('→')
            seniority_map = {'junior': 1, 'associate': 1, 'mid': 2, 'intermediate': 2, 'senior': 3, 'lead': 4, 'manager': 4, 'director': 5}
            progression_score, prev_level = 0, 0
            for role in roles:
                for title, level in seniority_map.items():
                    if title in role.lower():
                        if level > prev_level: progression_score += 1
                        prev_level = level; break
            profile['career_progression'] = progression_score / max(len(roles), 1)
        if 'avg_tenure_years' in candidate_data:
            profile['stability_score'] = min(candidate_data['avg_tenure_years'] / 5, 1.0)
        if 'platform_activity' in candidate_data:
            activity = str(candidate_data['platform_activity']).lower()
            activity_signals = ['blog', 'github', 'speaker', 'mentor', 'contributor', 'open source']
            profile['recent_activity_score'] = sum(1 for signal in activity_signals if signal in activity) / len(activity_signals)
        return profile
    
    def rank_candidates(self, job_description, candidates_df):
        job_reqs = self.extract_job_requirements(job_description)
        ranked_candidates = []
        
        for idx, candidate in candidates_df.iterrows():
            candidate_text = f"{candidate.get('skills', '')} {candidate.get('career_history', '')} {candidate.get('education', '')} {candidate.get('summary', '')}"
            
            # 1. Semantic Score
            job_embedding = self.semantic_model.encode([job_description])
            cand_embedding = self.semantic_model.encode([candidate_text])
            semantic_score = cosine_similarity(job_embedding, cand_embedding)[0][0]
            
            # 2. Skill Match
            candidate_skills = str(candidate.get('skills', '')).lower()
            skill_match = 0
            if job_reqs['skills']:
                skill_match = sum(1 for skill in job_reqs['skills'] if skill in candidate_skills) / len(job_reqs['skills'])
            
            # 3. Experience Match
            exp_match = min(candidate.get('experience_years', 0) / max(job_reqs['experience_years'], 1), 1.5) if job_reqs['experience_years'] > 0 else 0.7
            
            # 4. Activity Bonus
            profile_analysis = self.analyze_candidate_profile(candidate.to_dict())
            activity_bonus = (profile_analysis['career_progression'] * 0.4 + profile_analysis['stability_score'] * 0.3 + profile_analysis['recent_activity_score'] * 0.3)
            
            # Final Score (Weights)
            final_score = (semantic_score * 0.40 + skill_match * 0.25 + exp_match * 0.20 + activity_bonus * 0.15)
            
            ranked_candidates.append({
                'name': candidate.get('name', f'Candidate {idx}'),
                'final_score': round(final_score * 100, 2),
                'semantic_score': round(semantic_score * 100, 2),
                'skill_match': round(skill_match * 100, 2),
                'experience_match': round(exp_match * 100, 2),
                'activity_score': round(activity_bonus * 100, 2),
                'years_experience': candidate.get('experience_years', 0),
                'skills': candidate.get('skills', ''),
                'profile': candidate.to_dict()
            })
        ranked_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return ranked_candidates[:10]

def create_sample_data():
    sample_candidates = [
        {'name': 'Sarah Johnson', 'skills': 'Python, ML, TensorFlow, PyTorch, AWS, Docker, Kubernetes, SQL, Deep Learning', 'experience_years': 7, 'career_history': 'Junior Data Scientist → Data Scientist → Senior ML Engineer → Lead ML Engineer', 'avg_tenure_years': 2.5, 'platform_activity': 'Active GitHub contributor, Tech blog writer, Conference speaker'},
        {'name': 'Mike Chen', 'skills': 'Java, Spring Boot, Microservices, AWS, Docker, React, PostgreSQL, Kafka', 'experience_years': 5, 'career_history': 'Software Developer → Senior Developer → Tech Lead', 'avg_tenure_years': 1.8, 'platform_activity': 'GitHub projects, Stack Overflow contributor'},
        {'name': 'Emily Rodriguez', 'skills': 'Python, R, SQL, Tableau, Power BI, Statistical Analysis, A/B Testing', 'experience_years': 4, 'career_history': 'Data Analyst → Senior Data Analyst → Data Science Lead', 'avg_tenure_years': 2.0, 'platform_activity': 'Tech meetup organizer, Mentorship program leader'},
        {'name': 'Alex Thompson', 'skills': 'Python, JavaScript, React, Node.js, MongoDB, AWS, GraphQL, Docker, CI/CD', 'experience_years': 6, 'career_history': 'Junior Developer → Full Stack Developer → Senior Developer → Engineering Manager', 'avg_tenure_years': 3.0, 'platform_activity': 'Open source maintainer, Tech blog author, Conference speaker'},
        {'name': 'Maria Garcia', 'skills': 'Python, TensorFlow, NLP, BERT, GPT, Transformers, MLOps, AWS SageMaker', 'experience_years': 5, 'career_history': 'NLP Researcher → ML Engineer → Senior ML Engineer', 'avg_tenure_years': 2.2, 'platform_activity': 'ACL publications, NLP workshop organizer, Open source NLP tools'},
        {'name': 'Robert Taylor', 'skills': 'Python, Java, Spring, AWS, Azure, GCP, Kubernetes, Terraform, Jenkins, DevOps', 'experience_years': 10, 'career_history': 'System Admin → DevOps Engineer → Senior DevOps → Cloud Architect', 'avg_tenure_years': 2.5, 'platform_activity': 'Cloud architecture blog, AWS Community Builder, Tech webinar host'},
        {'name': 'Jennifer Lee', 'skills': 'Python, SQL, Machine Learning, Deep Learning, Data Engineering, Spark, Airflow, dbt', 'experience_years': 6, 'career_history': 'Data Engineer → ML Engineer → Senior Data & ML Engineer', 'avg_tenure_years': 2.0, 'platform_activity': 'Data engineering meetup organizer, Technical writer, Open source contributor'}
    ]
    return pd.DataFrame(sample_candidates)

def save_results(ranked_candidates, job_description, requirements):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"talent_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    results_df = pd.DataFrame([{'Rank': i+1, 'Name': c['name'], 'Overall_Score': c['final_score'], 'Semantic_Fit': c['semantic_score'], 'Skill_Match': c['skill_match'], 'Experience_Match': c['experience_match'], 'Activity_Score': c['activity_score'], 'Years_Experience': c['years_experience'], 'Skills': c['skills'][:150]+'...'} for i, c in enumerate(ranked_candidates)])
    results_df.to_csv(f"{output_dir}/candidate_rankings.csv", index=False)
    
    analysis = {'timestamp': timestamp, 'extracted_requirements': requirements, 'top_candidates': [{'rank': i+1, 'name': c['name'], 'scores': {'overall': c['final_score'], 'semantic': c['semantic_score'], 'skill': c['skill_match'], 'experience': c['experience_match'], 'activity': c['activity_score']}} for i, c in enumerate(ranked_candidates[:5])]}
    with open(f"{output_dir}/detailed_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2, cls=NpEncoder)
    
    return output_dir

def main():
    st.set_page_config(page_title="TalentSight AI", page_icon="🔍", layout="wide")
    
    st.markdown("""
        <style>
        .main-header { font-size: 2.8rem; font-weight: 700; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
        .stButton > button { width: 100%; font-size: 1.2rem; font-weight: bold; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; }
        .stButton > button:hover { color: white; opacity: 0.9; }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔍 TalentSight AI</h1>', unsafe_allow_html=True)
    st.markdown("### Beyond Keywords: AI-Powered Semantic Talent Discovery")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded_file = st.file_uploader("Upload Candidate Data (CSV)", type=['csv'])
        st.info("💡 **How it works:**\n1. Upload a CSV of candidates.\n2. Paste a Job Description.\n3. Click analyze. TalentSight AI semantically ranks your candidates.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📋 Job Description")
        example_jds = {
            "Select...": "",
            "Senior ML Engineer": """We are looking for a Senior Machine Learning Engineer with 5+ years of experience. Strong experience with PyTorch, AWS, Docker, and Kubernetes is required. You will lead a team building production ML pipelines. PhD preferred.""",
            "Full Stack Developer": """Seeking a Full Stack Developer with 4+ years building web apps. Proficiency in React, Node.js, and Python required. Experience with AWS, Docker, and microservices is essential."""
        }
        selected = st.selectbox("Example Job Descriptions:", list(example_jds.keys()))
        job_description = st.text_area("Paste your job description here:", value=example_jds[selected], height=250)
    
    with col2:
        st.subheader("🔍 Extracted Requirements")
        if job_description:
            ai = TalentSightAI()
            reqs = ai.extract_job_requirements(job_description)
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Required Experience", f"{reqs['experience_years']}+ years")
                st.metric("Seniority Level", reqs['seniority_level'])
            with col2b:
                st.metric("Skills Detected", len(reqs['skills']))
            st.markdown("**🔑 Key Skills:**")
            if reqs['skills']:
                st.success(", ".join(reqs['skills'][:8]))
            else:
                st.warning("Add more detail for better extraction.")
    
    st.markdown("---")
    st.header("🏆 Candidate Rankings")
    if st.button("🚀 Analyze & Rank Candidates"):
        if not job_description:
            st.error("Please enter a job description!")
        else:
            with st.spinner("TalentSight AI is semantically analyzing and ranking your candidates..."):
                candidates_df = pd.read_csv(uploaded_file) if uploaded_file else create_sample_data()
                if not uploaded_file: st.info("Using internal sample data. Upload your own CSV for real results.")
                ai = TalentSightAI()
                ranked = ai.rank_candidates(job_description, candidates_df)
                
                # Save & Download
                output_dir = save_results(ranked, job_description, reqs)
                st.success(f"✅ Analysis Complete!")
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    csv_data = pd.DataFrame([{'Rank': i+1, 'Name': c['name'], 'Overall_Score': c['final_score'], 'Semantic_Fit': c['semantic_score'], 'Skill_Match': c['skill_match'], 'Experience_Match': c['experience_match'], 'Activity_Score': c['activity_score']} for i, c in enumerate(ranked)]).to_csv(index=False)
                    st.download_button("📊 Download CSV Results", data=csv_data, file_name="talent_rankings.csv", mime="text/csv")
                with col_d2:
                    st.download_button("📄 Download JSON Data", data=json.dumps({'rankings': ranked}, indent=2, cls=NpEncoder), file_name="talent_data.json", mime="application/json")

                # Display Visuals
                if ranked:
                    fig = go.Figure()
                    top5 = ranked[:5]
                    fig.add_trace(go.Bar(x=[c['name'] for c in top5], y=[c['final_score'] for c in top5], marker_color=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']))
                    fig.update_layout(title="Top 5 Semantic Ranked Candidates", xaxis_title="Candidates", yaxis_title="Match Score (%)", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                    for i, c in enumerate(ranked[:5]):
                        with st.expander(f"#{i+1} {c['name']} - Score: {c['final_score']:.1f}%", expanded=(i==0)):
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.markdown(f"**📈 Experience:** {c['years_experience']} years")
                                st.markdown(f"**🛠 Skills:** {c['skills'][:200]}...")
                                
                                # --- FIX: Added Unique Key to Radar Chart here ---
                                scores_df = pd.DataFrame({
                                    'Metric': ['Semantic Fit', 'Skill Match', 'Experience', 'Activity'],
                                    'Score': [c['semantic_score'], c['skill_match'], c['experience_match'], c['activity_score']]
                                })
                                fig_radar = px.line_polar(
                                    scores_df, r='Score', theta='Metric',
                                    line_close=True, range_r=[0, 100]
                                )
                                fig_radar.update_traces(fill='toself')
                                st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{i}")
                                
                            with col_b:
                                if c['final_score'] > 85: st.success("🏆 Top Recommendation")
                                elif c['final_score'] > 70: st.info("👍 Strong Match")
                                else: st.warning("📋 Review Further")
if __name__ == "__main__":
    main()
