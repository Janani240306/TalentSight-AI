import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
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
stop_words = set(stopwords.words('english'))

# --- Custom JSON Encoder to Fix your TypeError ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class AIRecruiter:
    def __init__(self):
        """Initialize the AI Recruiter with semantic models"""
        with st.spinner('🔧 Loading AI models... This might take a minute on first run'):
            self.semantic_model = load_semantic_model()
            self.stop_words = stop_words
        
    def extract_job_requirements(self, job_description):
        """
        Extract key requirements from job description using NLP
        Returns structured requirements dictionary
        """
        requirements = {
            'skills': [],
            'experience_years': 0,
            'education': [],
            'soft_skills': [],
            'responsibilities': [],
            'domain_knowledge': [],
            'seniority_level': ''
        }
        
        # Extract years of experience
        exp_patterns = [
            r'(\d+)\+?\s*(?:years|yrs).*?(?:experience|exp)',
            r'(?:experience|exp).*?(\d+)\+?\s*(?:years|yrs)',
            r'(\d+)\+?\s*years'
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, job_description.lower())
            if match:
                requirements['experience_years'] = int(match.group(1))
                break
        
        # Extract skills (common technical skills)
        skill_keywords = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node',
            'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'aws', 'azure',
            'gcp', 'docker', 'kubernetes', 'ci/cd', 'git', 'rest api', 'graphql',
            'machine learning', 'deep learning', 'nlp', 'computer vision', 'ai',
            'data analysis', 'statistics', 'tableau', 'power bi', 'excel',
            'project management', 'agile', 'scrum', 'kanban', 'leadership',
            'communication', 'teamwork', 'problem solving', 'critical thinking'
        ]
        
        job_lower = job_description.lower()
        requirements['skills'] = [skill for skill in skill_keywords if skill in job_lower]
        
        # Extract education requirements
        education_keywords = ['bachelor', 'master', 'phd', 'mba', 'degree', 'b.tech', 'm.tech', 'b.e', 'm.e']
        requirements['education'] = [edu for edu in education_keywords if edu in job_lower]
        
        # Detect soft skills
        soft_skill_keywords = [
            'communication', 'leadership', 'teamwork', 'problem-solving',
            'adaptability', 'creativity', 'time management', 'critical thinking',
            'collaboration', 'initiative', 'mentoring', 'negotiation'
        ]
        requirements['soft_skills'] = [skill for skill in soft_skill_keywords if skill in job_lower]
        
        # Determine seniority level
        if any(word in job_lower for word in ['senior', 'lead', 'principal', 'architect']):
            requirements['seniority_level'] = 'Senior'
        elif any(word in job_lower for word in ['junior', 'associate', 'entry']):
            requirements['seniority_level'] = 'Junior'
        else:
            requirements['seniority_level'] = 'Mid-level'
            
        return requirements
    
    def analyze_candidate_profile(self, candidate_data):
        """
        Analyze a candidate's profile comprehensively
        """
        profile = {
            'years_of_experience': 0,
            'skill_diversity': 0,
            'career_progression': 0,
            'stability_score': 0,
            'skill_depth': defaultdict(int),
            'recent_activity_score': 0,
            'education_level': 0,
            'total_roles': 0
        }
        
        if 'experience_years' in candidate_data:
            profile['years_of_experience'] = candidate_data['experience_years']
        
        if 'skills' in candidate_data:
            skills = str(candidate_data['skills']).lower().split(',')
            profile['skill_diversity'] = len(set(skills))
            for skill in skills:
                profile['skill_depth'][skill.strip()] += 1
        
        if 'career_history' in candidate_data:
            history = str(candidate_data['career_history'])
            roles = history.split('→')
            profile['total_roles'] = len(roles)
            
            seniority_map = {
                'junior': 1, 'associate': 1,
                'mid': 2, 'intermediate': 2,
                'senior': 3, 'lead': 4,
                'manager': 4, 'director': 5
            }
            
            progression_score = 0
            prev_level = 0
            
            for role in roles:
                role_lower = role.lower()
                for title, level in seniority_map.items():
                    if title in role_lower:
                        if level > prev_level:
                            progression_score += 1
                        prev_level = level
                        break
            
            profile['career_progression'] = progression_score / max(len(roles), 1)
        
        if 'avg_tenure_years' in candidate_data:
            profile['stability_score'] = min(candidate_data['avg_tenure_years'] / 5, 1.0)
        
        if 'platform_activity' in candidate_data:
            activity = str(candidate_data['platform_activity']).lower()
            activity_signals = ['blog', 'github', 'speaker', 'mentor', 'contributor', 'open source']
            profile['recent_activity_score'] = sum(1 for signal in activity_signals if signal in activity) / len(activity_signals)
        
        if 'education' in candidate_data:
            edu = str(candidate_data['education']).lower()
            edu_levels = {'phd': 5, 'master': 4, 'mba': 4, 'bachelor': 3, 'associate': 2}
            for level, score in edu_levels.items():
                if level in edu:
                    profile['education_level'] = score
                    break
        
        return profile
    
    def calculate_semantic_similarity(self, job_description, candidate_text):
        """
        Calculate semantic similarity between job description and candidate profile
        """
        job_embedding = self.semantic_model.encode([job_description])
        candidate_embedding = self.semantic_model.encode([candidate_text])
        
        similarity = cosine_similarity(job_embedding, candidate_embedding)[0][0]
        return similarity
    
    def rank_candidates(self, job_description, candidates_df):
        """
        Main ranking algorithm combining multiple factors
        """
        job_reqs = self.extract_job_requirements(job_description)
        
        ranked_candidates = []
        
        for idx, candidate in candidates_df.iterrows():
            # 1. Semantic Understanding Score (40% weight)
            candidate_text = f"{candidate.get('skills', '')} {candidate.get('career_history', '')} {candidate.get('education', '')} {candidate.get('summary', '')}"
            semantic_score = self.calculate_semantic_similarity(job_description, candidate_text)
            
            # 2. Skill Match Score (25% weight)
            candidate_skills = str(candidate.get('skills', '')).lower()
            skill_match = 0
            if job_reqs['skills']:
                skill_match = sum(1 for skill in job_reqs['skills'] if skill in candidate_skills) / len(job_reqs['skills'])
            
            # 3. Experience Match Score (20% weight)
            exp_match = min(candidate.get('experience_years', 0) / max(job_reqs['experience_years'], 1), 1.5) if job_reqs['experience_years'] > 0 else 0.7
            
            # 4. Profile Completeness & Activity Score (15% weight)
            profile_analysis = self.analyze_candidate_profile(candidate.to_dict())
            
            activity_bonus = (
                profile_analysis['career_progression'] * 0.4 +
                profile_analysis['stability_score'] * 0.3 +
                profile_analysis['recent_activity_score'] * 0.3
            )
            
            # Weighted Final Score
            final_score = (
                semantic_score * 0.40 +
                skill_match * 0.25 +
                exp_match * 0.20 +
                activity_bonus * 0.15
            )
            
            # Apply seniority bonus if matching
            seniority_bonus = 1.0
            if job_reqs['seniority_level']:
                candidate_text_lower = str(candidate.get('career_history', '')).lower()
                if job_reqs['seniority_level'].lower() in candidate_text_lower:
                    seniority_bonus = 1.1
            
            final_score *= seniority_bonus
            
            ranked_candidates.append({
                'candidate_id': idx,
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
        
        # Sort by final score
        ranked_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return ranked_candidates[:10]

def create_sample_data():
    """Create realistic sample candidate data"""
    sample_candidates = [
        # ... (Your sample data stays exactly the same as before, I truncated it for brevity, but keep your full list here) 
    ]
    # Note: Since your prompt was huge, just keep your existing 10 sample candidates!
    return pd.DataFrame(sample_candidates)

def save_results(ranked_candidates, job_description, requirements):
    """Save analysis results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"recruitment_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save ranked candidates as CSV
    results_df = pd.DataFrame([{
        'Rank': i+1,
        'Name': c['name'],
        'Overall_Score': c['final_score'],
        'Semantic_Fit': c['semantic_score'],
        'Skill_Match': c['skill_match'],
        'Experience_Match': c['experience_match'],
        'Activity_Score': c['activity_score'],
        'Years_Experience': c['years_experience'],
        'Skills': c['skills']
    } for i, c in enumerate(ranked_candidates)])
    
    csv_path = f"{output_dir}/candidate_rankings.csv"
    results_df.to_csv(csv_path, index=False)
    
    # Save detailed analysis as JSON
    analysis = {
        'timestamp': timestamp,
        'job_description': job_description,
        'extracted_requirements': requirements,
        'top_candidates': [
            {
                'rank': i+1,
                'name': c['name'],
                'scores': {
                    'overall': c['final_score'],
                    'semantic_fit': c['semantic_score'],
                    'skill_match': c['skill_match'],
                    'experience_match': c['experience_match'],
                    'activity': c['activity_score']
                },
                'profile': c['profile']
            } for i, c in enumerate(ranked_candidates[:5])
        ]
    }
    
    json_path = f"{output_dir}/detailed_analysis.json"
    with open(json_path, 'w') as f:
        # --- FIX: Added cls=NpEncoder below to stop the error ---
        json.dump(analysis, f, indent=2, cls=NpEncoder)
    
    # Save HTML report
    html_report = generate_html_report(ranked_candidates, job_description, requirements)
    html_path = f"{output_dir}/recruitment_report.html"
    with open(html_path, 'w') as f:
        f.write(html_report)
    
    return output_dir

def generate_html_report(ranked_candidates, job_description, requirements):
    """Generate a beautiful HTML report"""
    # HTML generation code stays exactly the same as yours
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Recruiter - Candidate Ranking Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
            .candidate-card {{ border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 8px; background: #fafafa; }}
            .score {{ font-size: 24px; font-weight: bold; color: #667eea; }}
            .rank {{ font-size: 36px; font-weight: bold; color: #764ba2; }}
            .skills {{ color: #666; margin: 10px 0; }}
            .badge {{ display: inline-block; padding: 5px 10px; background: #667eea; color: white; border-radius: 15px; margin: 5px; font-size: 12px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #667eea; color: white; }}
            .recommendation {{ padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .top {{ background: #d4edda; color: #155724; }}
            .strong {{ background: #fff3cd; color: #856404; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 AI Recruiter Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime("%B %d, %Y at %H:%M")}</p>
            </div>
            
            <h2>📋 Job Requirements Extracted</h2>
            <p><strong>Required Experience:</strong> {requirements['experience_years']}+ years</p>
            <p><strong>Seniority Level:</strong> {requirements['seniority_level']}</p>
            <p><strong>Key Skills Required:</strong></p>
            <div>
                {''.join([f'<span class="badge">{skill}</span>' for skill in requirements['skills'][:10]])}
            </div>
            
            <h2>🏆 Top Candidates Ranking</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Candidate</th>
                    <th>Overall Score</th>
                    <th>Semantic Fit</th>
                    <th>Skill Match</th>
                    <th>Experience</th>
                    <th>Activity</th>
                </tr>
    """
    
    for i, candidate in enumerate(ranked_candidates[:10]):
        html += f"""
                <tr>
                    <td><strong>#{i+1}</strong></td>
                    <td><strong>{candidate['name']}</strong></td>
                    <td class="score">{candidate['final_score']}%</td>
                    <td>{candidate['semantic_score']}%</td>
                    <td>{candidate['skill_match']}%</td>
                    <td>{candidate['experience_match']}%</td>
                    <td>{candidate['activity_score']}%</td>
                </tr>
        """
    
    html += """
            </table>
            
            <h2>📊 Detailed Candidate Analysis</h2>
    """
    
    for i, candidate in enumerate(ranked_candidates[:5]):
        recommendation_class = "top" if candidate['final_score'] > 85 else "strong" if candidate['final_score'] > 70 else ""
        recommendation_text = "🏆 Top Recommendation - Interview Immediately" if candidate['final_score'] > 85 else "👍 Strong Candidate - Proceed to Screening" if candidate['final_score'] > 70 else "📋 Consider - Review Further"
        
        html += f"""
            <div class="candidate-card">
                <h3><span class="rank">#{i+1}</span> {candidate['name']}</h3>
                <div class="score">Overall Score: {candidate['final_score']}%</div>
                <div class="recommendation {recommendation_class}">{recommendation_text}</div>
                <p><strong>Years of Experience:</strong> {candidate['years_experience']} years</p>
                <div class="skills"><strong>Skills:</strong> {candidate['skills'][:200]}...</div>
                <p><strong>Score Breakdown:</strong></p>
                <ul>
                    <li>Semantic Understanding: {candidate['semantic_score']}%</li>
                    <li>Skill Match: {candidate['skill_match']}%</li>
                    <li>Experience Match: {candidate['experience_match']}%</li>
                    <li>Activity Score: {candidate['activity_score']}%</li>
                </ul>
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    return html

def main():
    st.set_page_config(
        page_title="AI Recruiter - Smart Candidate Ranking",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .score-card {
            padding: 20px;
            border-radius: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .candidate-card {
            padding: 15px;
            border-radius: 8px;
            background: #f8f9fa;
            margin: 10px 0;
            border-left: 5px solid #667eea;
        }
        .metric-highlight {
            font-size: 1.2rem;
            font-weight: 600;
            color: #667eea;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎯 AI-Powered Intelligent Recruiter</h1>', unsafe_allow_html=True)
    st.markdown("### Beyond Keywords: Semantic Understanding for Perfect Candidate Matching")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")
        
        st.subheader("📊 Ranking Weights")
        semantic_weight = st.slider("Semantic Understanding", 0.0, 1.0, 0.40, 0.05)
        skill_weight = st.slider("Skill Match", 0.0, 1.0, 0.25, 0.05)
        experience_weight = st.slider("Experience Match", 0.0, 1.0, 0.20, 0.05)
        activity_weight = st.slider("Profile Activity", 0.0, 1.0, 0.15, 0.05)
        
        # Normalize weights
        total_weight = semantic_weight + skill_weight + experience_weight + activity_weight
        if total_weight > 0:
            semantic_weight /= total_weight
            skill_weight /= total_weight
            experience_weight /= total_weight
            activity_weight /= total_weight
        
        st.markdown("---")
        st.subheader("📁 Data Input")
        uploaded_file = st.file_uploader("Upload Candidate Data (CSV)", type=['csv'])
        
        st.markdown("---")
        st.info("💡 **How it works:**\n\n"
                "1. **Semantic Understanding**: Uses AI to truly understand job requirements\n"
                "2. **Multi-dimensional Analysis**: Evaluates skills, experience, career growth\n"
                "3. **Behavioral Signals**: Considers platform activity and engagement\n"
                "4. **Smart Ranking**: Combines all factors for trustworthy shortlists")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Job Description")
        
        # Example job descriptions
        example_jds = {
            "Select an example...": "",
            "Senior ML Engineer": """We are looking for a Senior Machine Learning Engineer with 5+ years of experience in Python and deep learning frameworks. 
            The ideal candidate should have strong experience with TensorFlow or PyTorch, AWS cloud services, and deploying ML models to production. 
            Knowledge of NLP, computer vision, or recommendation systems is a plus. 
            You will lead a team of ML engineers and work closely with data scientists to build scalable ML pipelines.
            Required: Docker, Kubernetes, CI/CD experience. Master's or PhD preferred.""",
            
            "Full Stack Developer": """Seeking an experienced Full Stack Developer with 4+ years building web applications. 
            Strong proficiency in React, Node.js, and Python required. Experience with MongoDB, PostgreSQL, and AWS is essential.
            Must have experience with Docker containers and microservices architecture.
            Knowledge of GraphQL, TypeScript, and testing frameworks is highly valued.""",
            
            "Data Science Lead": """Looking for a Data Science Lead to drive our analytics initiatives. 
            Requires 6+ years of experience in data analysis, statistical modeling, and machine learning. 
            Proficiency in Python, R, and SQL is mandatory. Experience with Tableau, Power BI, and A/B testing frameworks.
            Strong communication and leadership skills required. MBA or Master's in a quantitative field preferred."""
        }
        
        selected_example = st.selectbox("Choose an example or write your own:", list(example_jds.keys()))
        
        job_description = st.text_area(
            "Paste the job description here:",
            value=example_jds[selected_example],
            height=250,
            placeholder="Enter detailed job description including required skills, experience, qualifications..."
        )
    
    with col2:
        st.subheader("🔍 Extracted Requirements")
        if job_description:
            ai_recruiter = AIRecruiter()
            requirements = ai_recruiter.extract_job_requirements(job_description)
            
            # Display extracted requirements
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Required Experience", f"{requirements['experience_years']}+ years")
                st.metric("Seniority Level", requirements['seniority_level'])
            with col2b:
                st.metric("Skills Detected", len(requirements['skills']))
                st.metric("Soft Skills", len(requirements['soft_skills']))
            
            st.markdown("**🔑 Key Skills:**")
            if requirements['skills']:
                skills_text = ", ".join(requirements['skills'][:8])
                st.info(skills_text)
            else:
                st.warning("Enter more detailed job description to extract skills")
    
    # Candidate Ranking Section
    st.markdown("---")
    st.header("🏆 Candidate Rankings")
    
    if st.button("🚀 Analyze & Rank Candidates", type="primary", use_container_width=True):
        if not job_description:
            st.error("Please enter a job description first!")
        else:
            with st.spinner("🧠 AI is analyzing candidates with deep understanding..."):
                # Load data
                if uploaded_file:
                    candidates_df = pd.read_csv(uploaded_file)
                else:
                    candidates_df = create_sample_data()
                    st.info("Using sample candidate database. Upload your own CSV for custom analysis.")
                
                # Initialize and rank
                ai_recruiter = AIRecruiter()
                
                # Get requirements
                requirements = ai_recruiter.extract_job_requirements(job_description)
                
                # Rank candidates
                ranked_candidates = ai_recruiter.rank_candidates(job_description, candidates_df)
                
                # Override weights based on user configuration
                for candidate in ranked_candidates:
                    candidate['final_score'] = round(
                        candidate['semantic_score'] * semantic_weight / 0.40 +
                        candidate['skill_match'] * skill_weight / 0.25 +
                        candidate['experience_match'] * experience_weight / 0.20 +
                        candidate['activity_score'] * activity_weight / 0.15
                    , 2)
                
                ranked_candidates.sort(key=lambda x: x['final_score'], reverse=True)
                
                # Save results
                output_dir = save_results(ranked_candidates, job_description, requirements)
                
                # Display success message with file locations
                st.success(f"✅ Analysis Complete! Results saved to: `{output_dir}/`")
                
                # Display saved files
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info("📊 **CSV Rankings**\n`candidate_rankings.csv`")
                with col2:
                    st.info("📄 **Detailed Analysis**\n`detailed_analysis.json`")
                with col3:
                    st.info("🌐 **HTML Report**\n`recruitment_report.html`")
                
                # Top candidates visualization
                if ranked_candidates:
                    # Score distribution chart
                    fig = go.Figure()
                    
                    top_5 = ranked_candidates[:5]
                    names = [c['name'] for c in top_5]
                    scores = [c['final_score'] for c in top_5]
                    
                    fig.add_trace(go.Bar(
                        x=names,
                        y=scores,
                        marker_color=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe'],
                        text=[f"{s:.1f}%" for s in scores],
                        textposition='auto',
                    ))
                    
                    fig.update_layout(
                        title="Top 5 Candidates - Overall Score",
                        xaxis_title="Candidates",
                        yaxis_title="Match Score (%)",
                        template="plotly_white",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed candidate cards
                st.subheader("📊 Detailed Candidate Profiles")
                
                for i, candidate in enumerate(ranked_candidates[:5]):
                    with st.expander(f"#{i+1} {candidate['name']} - Score: {candidate['final_score']:.1f}%", expanded=(i==0)):
                        col_a, col_b = st.columns([2, 1])
                        
                        with col_a:
                            st.markdown(f"**📈 Experience:** {candidate['years_experience']} years")
                            st.markdown(f"**🛠 Skills:** {candidate['skills'][:200]}...")
                            
                            # Score breakdown
                            scores_df = pd.DataFrame({
                                'Metric': ['Semantic Fit', 'Skill Match', 'Experience', 'Activity'],
                                'Score': [
                                    candidate['semantic_score'],
                                    candidate['skill_match'],
                                    candidate['experience_match'],
                                    candidate['activity_score']
                                ]
                            })
                            
                            fig_radar = px.line_polar(
                                scores_df, 
                                r='Score', 
                                theta='Metric',
                                line_close=True,
                                range_r=[0, 100]
                            )
                            fig_radar.update_traces(fill='toself')
                            st.plotly_chart(fig_radar, use_container_width=True)
                        
                        with col_b:
                            # Key strengths
                            st.markdown("**💪 Key Strengths:**")
                            strengths = []
                            if candidate['semantic_score'] > 80:
                                strengths.append("✅ Excellent overall fit")
                            if candidate['skill_match'] > 70:
                                strengths.append("🎯 Strong skill alignment")
                            if candidate['experience_match'] > 80:
                                strengths.append("⭐ Perfect experience match")
                            if candidate['activity_score'] > 60:
                                strengths.append("🔥 Highly active professional")
                            
                            for strength in strengths:
                                st.success(strength)
                            
                            # Recommendation
                            if candidate['final_score'] > 85:
                                st.markdown("### 🏆 Top Recommendation")
                                st.markdown("*Immediate interview recommended*")
                            elif candidate['final_score'] > 70:
                                st.markdown("### 👍 Strong Candidate")
                                st.markdown("*Proceed to screening*")
                            else:
                                st.markdown("### 📋 Consider")
                                st.markdown("*Review further before proceeding*")
                
                # Insights panel
                st.markdown("---")
                st.subheader("📈 Recruitment Insights")
                
                insight_col1, insight_col2, insight_col3 = st.columns(3)
                
                with insight_col1:
                    avg_score = np.mean([c['final_score'] for c in ranked_candidates])
                    st.metric("Average Candidate Score", f"{avg_score:.1f}%")
                
                with insight_col2:
                    high_potential = sum(1 for c in ranked_candidates if c['final_score'] > 80)
                    st.metric("High-Potential Candidates", high_potential)
                
                with insight_col3:
                    skill_gap = len(requirements.get('skills', [])) - len(ranked_candidates[0]['skills'].split(',')) if ranked_candidates else 0
                    st.metric("Skill Coverage", f"{max(0, 100 - skill_gap*10)}%")
                
                # Download buttons for all output files
                st.markdown("---")
                st.subheader("📥 Download Results")
                
                col_d1, col_d2, col_d3 = st.columns(3)
                
                with col_d1:
                    csv_data = pd.DataFrame([{
                        'Rank': i+1,
                        'Name': c['name'],
                        'Overall_Score': c['final_score'],
                        'Semantic_Fit': c['semantic_score'],
                        'Skill_Match': c['skill_match'],
                        'Experience_Match': c['experience_match'],
                        'Activity_Score': c['activity_score']
                    } for i, c in enumerate(ranked_candidates)]).to_csv(index=False)
                    
                    st.download_button(
                        label="📊 Download CSV",
                        data=csv_data,
                        file_name="candidate_rankings.csv",
                        mime="text/csv"
                    )
                
                with col_d2:
                    json_data = json.dumps({
                        'rankings': [
                            {
                                'rank': i+1,
                                'name': c['name'],
                                'scores': {
                                    'overall': c['final_score'],
                                    'semantic': c['semantic_score'],
                                    'skill': c['skill_match'],
                                    'experience': c['experience_match'],
                                    'activity': c['activity_score']
                                }
                            } for i, c in enumerate(ranked_candidates)
                        ]
                    }, indent=2, cls=NpEncoder) # Added cls=NpEncoder here too just in case
                    
                    st.download_button(
                        label="📄 Download JSON",
                        data=json_data,
                        file_name="detailed_analysis.json",
                        mime="application/json"
                    )
                
                with col_d3:
                    html_report = generate_html_report(ranked_candidates, job_description, requirements)
                    st.download_button(
                        label="🌐 Download HTML Report",
                        data=html_report,
                        file_name="recruitment_report.html",
                        mime="text/html"
                    )

if __name__ == "__main__":
    main()