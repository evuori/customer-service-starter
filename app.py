import streamlit as st
from customer_support_bot import part_1_graph
import uuid

st.set_page_config(
    page_title="Customer Service",
    page_icon="✈️",
    layout="wide"
)

st.title("Customer Service")
st.subheader("Extrene Travels Customer Service")

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "passenger_id" not in st.session_state:
    st.session_state.passenger_id = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    passenger_id = st.text_input("Enter Passenger ID:", 
                                value=st.session_state.passenger_id if st.session_state.passenger_id else "",
                                placeholder="8149 604011")
    
    if passenger_id:
        st.session_state.passenger_id = passenger_id
    
    st.markdown("""
    ### Sample Questions
    - What time is my flight?
    - Can I change my flight?
    - What's your cancellation policy?
    - Are there any hotels near the airport?
    - Can you recommend activities in Zurich?
    """)
    
    st.markdown("""
    ### Example Customers
    - 8149 604011
    - 8149 604012
    - 8149 604013
    - 8149 604014
    - 8149 604015
    - 8149 604016
    - 8149 604017
    """)

# Main chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you today?"):
    if not st.session_state.passenger_id:
        st.error("Please enter a Passenger ID in the sidebar first!")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Working on it..."):
                config = {
                    "configurable": {
                        "passenger_id": st.session_state.passenger_id,
                        "thread_id": st.session_state.thread_id
                    }
                }
                messages = [("user", prompt)]
                result = part_1_graph.invoke({"messages": messages}, config=config)
                
                # Extract the assistant's response
                assistant_message = result["messages"][-1].content
                st.markdown(assistant_message)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": assistant_message})

# Footer
st.markdown("---")
st.markdown("*Powered by Extreme Travels AI*") 