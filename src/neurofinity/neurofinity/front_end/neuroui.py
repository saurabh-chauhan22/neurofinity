import streamlit as st
import time

from backend.audio_to_text import AudioToText
from backend.nlp import NLPTransacripter

def main():
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "recording" not in st.session_state:
        st.session_state.recording = False  # Track recording state
    
    if st.session_state.page == "home":
        show_home_page()
    elif st.session_state.page == "process_audio":
        process_audio()
    elif st.session_state.page == "mind_map":
        generate_mind_map()

def show_home_page():
    st.image("background.jpg", use_container_width=True)
    st.markdown("<h1 style='text-align: center;'>Neuroinfy</h1>", unsafe_allow_html=True)
    
    st.subheader("Upload an Audio File")
    uploaded_file = st.file_uploader("Choose an MP3 or WAV file", type=["mp3", "wav"])
    audio_to_text_instance = AudioToText(audio_path=uploaded_file)
    audio_file_transcript = audio_to_text_instance.output_text()
    transcript_to_summary = NLPTransacripter(transcript=audio_file_transcript)
    output_dict = transcript_to_summary.task_extraction()
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/mp3' if uploaded_file.type == 'audio/mpeg' else 'audio/wav')
        st.success("File uploaded successfully!")
        st.session_state.page = "process_audio"
        st.rerun()
    
    st.subheader("Record Your Voice")
    
    # Custom CSS for circular and larger buttons
    st.markdown(
        """
        <style>
        .stButton button {
            border-radius: 50%;
            width: 150px;
            height: 150px;
            font-size: 24px;
            margin: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Layout for Start and Stop buttons (side by side)
    col1, col2 = st.columns(2)  # Two columns for side-by-side buttons
    
    with col1:
        if st.button("🎤 Start", key="start_recording"):
            st.session_state.recording = True
            st.write("Recording started... Speak now!")
    
    with col2:
        if st.button("⏹️ Stop", key="stop_recording"):
            if st.session_state.recording:
                st.session_state.recording = False
                st.success("Recording stopped and saved successfully!")
                st.session_state.page = "process_audio"
                st.rerun()
            else:
                st.warning("Recording is not in progress.")

def process_audio():
    st.write("Processing audio and extracting summary...")
    time.sleep(2)  # Simulated processing time
    
    # Simulated summary extraction
    summary = {"Topic": "Task Management", "Key Points": ["Break tasks into steps", "Set reminders", "Use visuals"]}
    st.session_state.summary = summary
    
    st.session_state.page = "mind_map"
    st.rerun()

def generate_mind_map():
    summary = st.session_state.get("summary", {})
    if not summary:
        st.write("Mind map generation failed. No summary available.")
        return
    
    # Define colors
    main_topic_color = "#FF6F61"  # Main topic color
    subtopic_color = "#6B5B95"    # Subtopic color
    background_color = "#F0F2F6"  # Background color
    
    # Create the Graphviz graph
    graph = f"""
    digraph G {{
        bgcolor="{background_color}";  // Set background color
        fontname="Arial";  // Set font style
        node [fontname="Arial", fontsize="20"];  // Set font size for nodes
        edge [arrowhead=normal];  // Use normal arrows
        
        // Main topic (smaller curved-edge square, bold text)
        "{summary["Topic"]}" [shape=box, style="filled,rounded", fillcolor="{main_topic_color}", fontsize="24", fontname="Arial-Bold", penwidth=2, width=1, height=0.5];
        
        // Subtopic nodes (curved-edge rectangles)
    """
    
    # Add subtopics
    for point in summary["Key Points"]:
        graph += f'"{point}" [shape=box, style="filled,rounded", fillcolor="{subtopic_color}", fontsize="20"];\n'
        graph += f'"{summary["Topic"]}" -> "{point}" [color="{subtopic_color}", penwidth=2];\n'
    
    graph += "}}"
    
    st.graphviz_chart(graph)

def switch_page(page_name):
    st.session_state.page = page_name
    st.rerun()
    

if __name__ == "__main__":
    main()
