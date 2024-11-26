import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
import utils
import re

class YouTubeTranscriptChat:
    def __init__(self):
        self.initialize_session_state()
        self.llm = utils.configure_llm()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'youtube_chat_history' not in st.session_state:
            st.session_state['youtube_chat_history'] = []
        if 'youtube_transcript' not in st.session_state:
            st.session_state['youtube_transcript'] = ""
            
    def get_video_id(self, youtube_url):
        """Extract video ID from YouTube URL"""
        video_id = None
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
            r'(?:embed\/)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                video_id = match.group(1)
                break
                
        return video_id

    def get_transcript(self, video_id):
        """Get transcript from YouTube video"""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = ' '.join([t['text'] for t in transcript_list])
            return transcript
        except Exception as e:
            st.error(f"Error fetching transcript: {str(e)}")
            return None

    def create_conversation_chain(self, transcript):
        """Create a conversation chain with the transcript"""
        # Split the transcript into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(transcript)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = DocArrayInMemorySearch.from_texts(chunks, embeddings)

        # Create memory and conversation chain
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
        )

        return conversation_chain

    def process_input(self, user_question, conversation_chain):
        """Process user input and get response"""
        response = conversation_chain({'question': user_question})
        return response['answer']

    def render_chat(self):
        """Render the chat interface"""
        for message in st.session_state['youtube_chat_history']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def main(self):
        st.title("💬 Chat with YouTube Video")
        
        # YouTube URL input
        youtube_url = st.text_input("Enter YouTube Video URL:")
        
        if youtube_url:
            video_id = self.get_video_id(youtube_url)
            
            if video_id:
                if 'conversation_chain' not in st.session_state or st.session_state['youtube_transcript'] != youtube_url:
                    transcript = self.get_transcript(video_id)
                    if transcript:
                        st.session_state['conversation_chain'] = self.create_conversation_chain(transcript)
                        st.session_state['youtube_transcript'] = youtube_url
                        st.session_state['youtube_chat_history'] = []
                        st.success("Transcript loaded successfully! You can now chat about the video content.")
            else:
                st.error("Invalid YouTube URL. Please enter a valid URL.")

        # Chat interface
        self.render_chat()

        if user_question := st.chat_input("Ask about the video content"):
            if 'conversation_chain' in st.session_state:
                # Add user message to chat history
                st.session_state['youtube_chat_history'].append({"role": "user", "content": user_question})
                
                with st.chat_message("user"):
                    st.markdown(user_question)

                # Get bot response
                with st.chat_message("assistant"):
                    response = self.process_input(user_question, st.session_state['conversation_chain'])
                    st.markdown(response)
                    
                # Add assistant response to chat history
                st.session_state['youtube_chat_history'].append({"role": "assistant", "content": response})
            else:
                st.warning("Please enter a YouTube URL first.")

if __name__ == "__main__":
    obj = YouTubeTranscriptChat()
    obj.main()
