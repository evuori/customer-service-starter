# Customer Service Starter

An intelligent customer service AI tool that helps customers with flight information, bookings, and travel-related queries. The bot uses Claude 3 Sonnet for natural language understanding and Azure OpenAI for embeddings. Langgraph framwork is used for state and agent chaining & routing. This starter is based on official Langgraph documentation.

## Features

- Flight information lookup
- Ticket management (view, update, cancel)
- Car rental services
- Hotel bookings
- Trip recommendations
- Policy information retrieval
- Natural language conversation

## Prerequisites

- Python 3.8+
- Anthropic API key (for Claude 3)
- Azure OpenAI API key and endpoint (for embeddings)
- SQLite database with travel information

## Environment Variables

Create a `.env` file in the project root with the following variables:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_VERSION=your_api_version
DB_URL=your_database_url
FAQ_URL=your_faq_url
MODEL_NAME=claude-3-sonnet-20240229  # Optional, defaults to this value
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com:evuori/customer-service-starter.git
cd swiss-airlines-bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the bot:
```bash
python customer_support_bot.py
```

2. The bot will automatically:
   - Download and set up the SQLite database
   - Load FAQ information
   - Initialize the conversation system
   - Start processing customer queries

3. Example interaction:
```python
config = {"configurable": {"passenger_id": "0123456789"}}
messages = [("user", "Hi there, what time is my flight?")]
result = part_1_graph.invoke({"messages": messages}, config=config)
```

## Project Structure

- `customer_support_bot.py`: Main bot implementation
- `travel2.sqlite`: SQLite database (downloaded automatically)
- `.env`: Environment variables (not included in repository)
- `requirements.txt`: Project dependencies

## Logging

The bot includes comprehensive logging that shows:
- Conversation flow
- Tool execution
- State transitions
- Error messages
- Agent and tool names

Logs are written to the console with timestamps and log levels.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Anthropic for Claude 3
- Microsoft Azure for OpenAI services
- LangChain for the framework