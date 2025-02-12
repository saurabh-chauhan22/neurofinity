from streamlit import markdown, graphviz_chart
import streamlit as st
import time

st.image("background.jpg",use_container_width=True)

markdown(
    """
    <style>
        /* Set the full-page background image by targeting the body element */
        body {
            background-size: cover; /* Cover the entire page */
            background-position: center; /* Center the background image */
            background-repeat: no-repeat; /* Do not repeat the image */
            background-attachment: fixed; /* Fix the background image */
        }

        /* Style for the header: full-width, fixed at the top, and centered */
        .header {
            background-color: #a3c6c4;
            padding: 20px;
            text-align: center;
            width: 100%;
            position: fixed;
            top: 30px;
            left: 0;
            z-index: 1000;
        }

        /* Add top padding to the content to avoid being hidden under the fixed header */
        .content {
            padding-top: 90px; /* Adjust as needed based on header height */
        }

        /* Ensure the content is readable over the background image */
        .content, .graphviz-container, footer {
            background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent white background */
            padding: 10px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
markdown(
    """
    <div class="header">
        <h1 style="color: #0d0c0c;">Neuroinfy</h1>
    </div>
    """,
    unsafe_allow_html=True
)

markdown('<div class="content">', unsafe_allow_html=True)

markdown(
    """
    <div style="background-color: #fff8f0; padding: 5px; margin-top: 5px;">
        <h2 style="color: #0d0c0c; text-align: center;">Upload an Audio File</h2>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Choose an MP3 or WAV file", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(
        uploaded_file,
        format='audio/mp3' if uploaded_file.type == 'audio/mpeg' else 'audio/wav'
    )
    st.success("File uploaded successfully!")

st.subheader("Record Your Voice")

if st.button("🎤", key="record", help="Click to start recording"):
    with st.spinner('Recording... Speak now!'):
        time.sleep(3)
    st.success("Recording saved successfully!")





# Close the container div
markdown('</div>', unsafe_allow_html=True)

markdown(
    """
    <footer style="text-align:center; padding: 20px; color: #666;">
        <p>Neuroinfy | Designed to help neurodivergent individuals stay organized and focused.</p>
    </footer>
    """,
    unsafe_allow_html=True
)

markdown('</div>', unsafe_allow_html=True)

st.set_option("client.showErrorDetails", True)